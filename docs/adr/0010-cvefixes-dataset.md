# ADR-0010: Dataset do GVF a partir do CVEfixes

- Status: Aceita
- Data: 2026-10-25

## Contexto

A 1.1.0 mostrou, com validação por grupo, que o GVF decora o dataset
sintético em vez de aprender. O bloqueio para um detector útil é dado real
rotulado em Python para VC-1, VC-2 e VC-3. Os datasets clássicos de
vulnerabilidade (Big-Vul, Devign) são de C/C++.

O **CVEfixes** coleta automaticamente, a partir do NVD, os commits que
corrigiram CVEs em projetos de código aberto, com o código antes e depois
de cada arquivo alterado. A versão 1.0.8 (julho de 2024, licença CC BY 4.0)
cobre 11.873 CVEs, 12.107 commits, 51.342 arquivos e 4.249 projetos, em um
dump SQL de 12,7 GB compactado.

## Decisão

1. **Fonte inicial: CVEfixes 1.0.8.** O download e a preparação rodam no
   Google Colab, não na máquina de referência (3,8 GB de RAM).
2. **Mapeamento de CWE para as categorias do MVP:**
   - VC-1 (injeção SQL/comando): CWE-89, CWE-77, CWE-78, CWE-88.
   - VC-2 (deserialização insegura): CWE-502.
   - VC-3 (segredos no código): CWE-798, CWE-259, CWE-321.
   A tabela fica no código, num único lugar, e é impressa no relatório.
3. **Rótulos por par antes/depois.** O arquivo antes da correção recebe a
   categoria da CWE do CVE. O arquivo depois da correção é um **negativo
   difícil**: quase idêntico, sem a falha corrigida. Ressalva assumida: o
   arquivo corrigido pode conter outras falhas que o CVE não cobre (ruído de
   rótulo).
4. **`group` = repositório.** O par antes/depois e todo o código do mesmo
   projeto ficam inteiros no treino ou na validação (ADR-0009, item 6).
5. **Limpeza**, com cada descarte contado no relatório: arquivos que não
   são Python, sem código antes (arquivos criados na correção), que o parser
   do GVF não lê (inclui Python 2, contado à parte por heurística), acima de
   um limite de linhas, sem mudança real entre antes e depois, e conteúdo
   duplicado (SHA-256), por exemplo em forks.
6. **Censo antes de construir.** O primeiro passo só mede: quantos pares
   utilizáveis existem por categoria, de quantos repositórios, o tamanho dos
   grafos resultantes e quais outras CWEs aparecem em Python (para escolher
   as categorias da v2). Só com esses números decide-se se o volume basta ou
   se é preciso complementar, por exemplo com avisos de segurança de pacotes
   PyPI que apontam o commit da correção.
7. **Linhas alteradas.** O censo registra se o diff da correção está
   disponível, para que o dataset final possa medir se a localização
   apontada pelo modelo cai perto da linha corrigida.
8. **O dataset não entra no repositório.** O código vem de milhares de
   projetos, cada um com a própria licença. O repositório guarda só as
   ferramentas que baixam e montam o dataset localmente. Os resultados
   publicados citam o CVEfixes conforme a CC BY 4.0.
9. **Integridade.** O preparo confere o MD5 publicado no Zenodo antes de
   usar o arquivo e verifica o esquema do banco antes de consultá-lo: uma
   coluna ausente gera erro com o esquema real, nunca um censo vazio
   silencioso.

## Alternativas consideradas

1. **Gerar vulnerabilidades sinteticamente** (injetar padrões em código
   limpo) — rejeitada como fonte principal: o modelo aprenderia os
   artefatos do gerador, o mesmo problema dos templates da 1.1.0.
2. **Usar regras de SAST (por exemplo, Bandit) como rótulo** — rejeitada:
   o modelo aprenderia a imitar as regras, inclusive os falsos positivos.
3. **Rotular por função em vez de arquivo** — adiada: o CVEfixes tem a
   tabela de métodos, mas o GVF analisa arquivos inteiros. Pode virar um
   segundo nível de granularidade depois.

## Consequências

- Positivo: rótulos vêm de correções reais de CVEs, com negativos difíceis
  de graça e agrupamento natural por projeto.
- Negativo: o volume em Python para três categorias pode ser pequeno; o
  censo existe para medir isso antes de qualquer investimento maior.
- Risco: ruído de rótulo (o "depois" nem sempre está limpo; CWE mal
  classificada no NVD). Mitigação: relatório de descartes e, no dataset
  final, uma amostra para revisão manual.
