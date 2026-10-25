# ADR-0007: Raciocínio latente iterativo no GSDM

- Status: Aceita
- Data: 2026-10-25

## Contexto

LLMs transformer modernos ganharam capacidade de "raciocínio" através de
chain-of-thought: o modelo gera passos intermediários em texto antes da
resposta final, o que melhora a qualidade da decisão em problemas
complexos.

O GSDM (ADR-0001) explicitamente não gera texto — é um modelo de decisão,
não de conversação. Isso não significa, porém, que ele deva decidir em um
único passo direto (entrada → score). Para análises de vulnerabilidade que
dependem de rastrear fluxo de dados através de múltiplas funções/escopos
(ex: VC-1, onde o dado precisa ser rastreado da entrada até um sink distante
no grafo), um único passe de agregação pode não ser suficiente para
capturar relações de longo alcance no grafo.

Precisamos de uma forma de "pensar mais" sobre casos difíceis, sem violar
o requisito central do GSDM de nunca gerar linguagem (NG2 do SPEC.md).

## Decisão

O GSDM incorpora **raciocínio latente iterativo**: a cabeça de decisão não
consome o embedding de um único passe de message-passing da GNN (ADR-0002).
Em vez disso, o encoder executa **N rodadas** de refinamento sobre o
mesmo grafo — cada rodada atualiza o estado latente de cada nó com base no
estado dos nós vizinhos (message-passing), permitindo que informação de
fluxo de dados "viaje" por mais hops do grafo a cada rodada.

- O número de rodadas N pode ser fixo (hiperparâmetro de treino) ou
  **adaptativo**: um pequeno classificador auxiliar decide, a cada rodada,
  se o estado já convergiu o suficiente para decidir, ou se vale mais uma
  rodada (mecanismo análogo a "pensar mais" em problemas difíceis,
  inteiramente em espaço latente — nenhum texto é gerado em nenhum momento).
- O estado latente nunca é decodificado para texto/linguagem em nenhuma
  etapa intermediária. Apenas o estado final, após a última rodada,
  alimenta a cabeça de decisão (ADR-0001) que emite os scores calibrados.

Isso é o equivalente estrutural de chain-of-thought: múltiplos passos de
computação antes da resposta, mas os "passos" são atualizações de estado
latente sobre o grafo do código, não frases.

## Alternativas consideradas

1. **Passe único de agregação (sem iteração)** — mais simples e mais
   rápido, mas insuficiente para capturar dependências de fluxo de dados
   de longo alcance no grafo (ex: dado atravessando 5+ funções antes de
   chegar a um sink). Rejeitado como abordagem única; fica como baseline
   de comparação (N=1) nos experimentos.
2. **Chain-of-thought textual tradicional** — geraria explicações em texto
   entre os passos. Rejeitado diretamente: contradiz NG2 e RF-5 do SPEC —
   o GSDM nunca gera linguagem, em nenhuma etapa, intermediária ou final.
3. **Número fixo de rodadas sem mecanismo adaptativo** — mais simples de
   implementar e treinar; fica como primeira implementação válida do MVP.
   O mecanismo adaptativo (rodadas variáveis por dificuldade do caso) é
   uma melhoria de v2, não bloqueia o MVP.

## Consequências

- Positivo: modelo pode capturar relações estruturais de longo alcance no
  grafo sem jamais gerar texto, preservando a identidade central do GSDM.
- Positivo: abre caminho natural para melhorias futuras (rodadas
  adaptativas) sem mudar a arquitetura de camadas (Clean Architecture,
  ADR-0003) — é uma mudança interna ao `infrastructure/model/`.
- Negativo: N rodadas multiplicam o custo computacional de inferência por N
  — precisa ser medido contra a meta de latência (RNF-4) antes de fixar N
  para o MVP.
- Risco: número fixo de rodadas mal escolhido pode causar
  over-smoothing (nós do grafo convergindo para representações
  indistinguíveis) — mitigação: validar empiricamente N no conjunto de
  validação antes de travar o valor para o MVP, e considerar técnicas
  conhecidas de GNN para mitigar over-smoothing (ex: conexões residuais
  entre rodadas) se o problema aparecer.
