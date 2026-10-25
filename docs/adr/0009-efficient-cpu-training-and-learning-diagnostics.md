# ADR-0009: Treino eficiente em CPU e diagnóstico de aprendizado (1.1.0)

- Status: Aceita
- Data: 2026-10-25

## Contexto

A 1.0.0 mostrou dois problemas medidos, não supostos:

1. **Desempenho.** O treino gastava ~3 ms por grafo em overhead do Python
   por operação, não em matemática. Cada grafo passava sozinho pelo modelo.
2. **Aprendizado ilusório.** Um GVF treinado com arquivos sintéticos acertou
   o próprio treino e errou um `pickle.loads` inédito. A validação sorteava
   arquivos individuais, então variações do mesmo exemplo caíam no treino e
   na validação ao mesmo tempo: a validação dizia "aprendeu" quando o modelo
   tinha decorado.

A máquina de referência para "computador fraco" é a do autor do projeto:
Intel Celeron N4100 (4 núcleos, nível x86-64-v2: SSE4.2, **sem AVX2/FMA**),
3,8 GB de RAM, vídeo integrado Intel UHD 600, sem GPU NVIDIA.

## Decisão

### Desempenho em CPU

1. **Treino em lote.** Vários grafos são unidos num único grafo disjunto
   (índices de nós deslocados, um segmento por grafo). Um passo de treino
   faz um forward e um backward para o lote inteiro, dividindo o overhead
   do Python pelo tamanho do lote. O readout passa a ser por segmento:
   softmax de atenção e soma restritas aos nós de cada grafo.
2. **Readout equivalente, só com matrizes 2-D.** `logit[g,c] = Σ_{i∈g}
   α[i,c]·(h[i]·w_c) + b_c`, que é matematicamente idêntico ao pooling da
   1.0.0 (pooling seguido de produto escalar é linear). Os nomes e formatos
   dos parâmetros não mudam: modelos da 1.0.0 continuam carregando.
3. **float32 como padrão.** Metade da memória e mais rápido em CPU fraca.
   float64 continua disponível (`gsdm.autograd.default_dtype`) e é usado
   nos testes de gradiente, onde a precisão numérica importa.
4. **Propagação vetorizada.** O `np.add.at` da 1.0.0 é substituído por um
   plano de soma pré-computado por relação (arestas ordenadas uma vez +
   `np.add.reduceat`), em ambos os sentidos (forward e backward).
5. **Sem assembly próprio.** As multiplicações de matrizes já são feitas
   pelo BLAS do NumPy (OpenBLAS), com kernels em assembly escolhidos em
   tempo de execução para cada CPU. Na máquina de referência, um kernel
   AVX2 escrito à mão travaria com "instrução ilegal". Código de baixo
   nível próprio só entra se um profiler apontar um gargalo que o NumPy não
   resolve, e então como kernel em C com detecção de CPU.

### Diagnóstico de aprendizado

6. **Validação por grupo.** O manifesto de treino aceita um campo opcional
   `group` (template, projeto ou CVE de origem). Um grupo inteiro vai só
   para o treino ou só para a validação. Sem `group`, cada arquivo é o
   próprio grupo, e o resumo do treino avisa que a validação pode estar
   contaminada por quase-duplicatas.
7. **Logs por época, no estilo de treino de LLM:** loss de treino, loss de
   validação, a diferença entre as duas (*generalization gap*), precisão,
   recall e F1 por decisão, erro de calibração, tempo e grafos/segundo.
   Saem no terminal e em `metrics.jsonl` (uma linha JSON por época).
8. **Painel gráfico.** Um único `dashboard.html`, sem dependências nem
   recursos externos, com as curvas em SVG. É reescrito a cada época, então
   pode ser aberto durante o treino.
9. **Diagnóstico automático** ao fim do treino: `learning`, `memorizing`,
   `stalled` ou `no-validation`. As regras são heurísticas documentadas no
   código (por exemplo: a loss de validação subiu desde a melhor época
   enquanto a de treino continuou caindo). É um alerta, não uma prova.
10. **Checkpoints e early stopping.** O checkpoint guarda pesos, melhores
    pesos, estado do Adam, estado do gerador aleatório e histórico, de modo
    que um treino interrompido e retomado é **bit a bit idêntico** a um
    treino contínuo (verificado por teste). O treino para quando a
    validação não melhora por N épocas, e o modelo final usa os **melhores
    pesos de validação**, não os últimos. A retomada recusa um manifesto
    diferente do original (verificado por SHA-256).

### Escala: o que a GSDM promete e o que não promete

11. O treino informa antes de começar a contagem de parâmetros e a memória
    estimada (~16 bytes por parâmetro em treino: peso, gradiente e dois
    momentos do Adam, mais ativações).
12. O passo de raciocínio tem pesos compartilhados: mais rodadas de
    raciocínio aumentam o custo de computação, **não** o número de
    parâmetros. Os parâmetros crescem com `(relações + 3) · hidden²`.
13. A 1.1.0 treina em **um processo, em CPU**. Na máquina de referência
    (3,8 GB), o limite prático de treino fica na casa das dezenas de
    milhões de parâmetros. **Bilhões e trilhões de parâmetros estão fora do
    escopo**: exigiriam GPUs e treino distribuído com fragmentação de
    pesos. Os lotes por segmento são compatíveis com paralelismo de dados
    no futuro, mas isso não é uma promessa desta versão.

## Alternativas consideradas

1. **GPU first (CuPy/CUDA)** — adiada: não há GPU NVIDIA na máquina de
   referência nem na CI do GitHub, então não haveria como verificar.
2. **Numba / extensão em C** — adiadas: o gargalo medido (overhead por
   operação) se resolve com lotes; dependências compiladas pesam na
   instalação e o Numba tende a demorar para suportar versões novas do
   Python.
3. **Validação aleatória por arquivo** — rejeitada como padrão: foi a causa
   direta do aprendizado ilusório da 1.0.0.

## Consequências

- Positivo: treino várias vezes mais rápido na mesma máquina (números na
  seção de benchmark do README), metade da memória, e um sinal objetivo de
  "aprendeu ou decorou".
- Positivo: compatível com a 1.0.0 — `model(graph)`, `Trainer.fit` e os
  arquivos de modelo continuam funcionando.
- Negativo: mais código de orquestração de treino (sessão, checkpoints,
  relatórios), todo coberto por testes.
- Risco: float32 acumula erro numérico diferente de float64; mitigado
  mantendo os testes de gradiente em float64 e um teste de invariância
  "lote = grafos individuais" com tolerância de float32.
