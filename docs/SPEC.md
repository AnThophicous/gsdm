# Especificação — GVF (Graph-based Vulnerability Finder)

> Status: DRAFT — em revisão. Nenhuma linha de código de produção deve ser
> escrita antes desta spec ser aprovada.
>
> Formato: Spec Kit (Problem Statement → Requirements → Non-Goals → ADRs →
> Acceptance Criteria), conforme decidido em 2026-10-25.

## 1. Problem Statement

Revisores humanos de código não escalam: não conseguem analisar todo commit,
todo PR, todo arquivo de um projeto grande com atenção constante a padrões de
vulnerabilidade. Ferramentas de SAST (static analysis) tradicionais existem,
mas são baseadas em regras fixas (regex/padrões), com alto índice de falsos
positivos e sem noção de contexto do projeto.

Precisamos de um sistema que:
- entenda o **contexto de um projeto** (não apenas um arquivo isolado),
- analise esse contexto em busca de **vulnerabilidades conhecidas**,
- **não converse** — não gera texto livre, apenas **decide** e retorna uma
  estrutura tipada com grau de confiança por achado,
- rode **localmente** na máquina do usuário, ou remotamente numa VPS exposta
  via MCP.

## 2. Goals (o que este projeto é)

- G1. Um motor de ML com arquitetura própria (não-transformer, não-LLM
  generativo) especializado em classificar trechos/grafos de código quanto a
  categorias de vulnerabilidade pré-definidas.
- G2. Saída estruturada em **XML**, com schema fixo e versionado, contendo
  categoria, confiança (probabilidade calibrada) e localização exata
  (arquivo:linha:coluna) de cada achado.
- G3. Sistema de confiança **calibrado** — 80% de confiança deve corresponder
  empiricamente a ~80% de acerto no conjunto de validação, não um número
  arbitrário.
- G4. Execução local via leitura direta de filesystem, e execução remota via
  um servidor MCP que expõe uma tool de análise.
- G5. Base de código em Python, organizada estritamente segundo Clean
  Architecture (Robert C. Martin): regra de dependência unidirecional,
  entidades de domínio sem dependência de frameworks, casos de uso isolados
  de infraestrutura.
- G6. Histórico de Git seguindo as convenções de Pro Git: commits atômicos,
  mensagens no formato Conventional Commits, branches de feature de vida
  curta.

## 3. Non-Goals (o que este projeto explicitamente NÃO é)

- NG1. Não é um concorrente direto do Jev (TypeSafe AI) nem uma cópia da
  arquitetura "System One" deles — a semelhança é apenas conceitual (saída
  estruturada e não-conversacional). Nossa categoria de arquitetura tem
  nome próprio: **GSDM** (ver Glossário, seção 7).
- NG2. Não é um LLM generalista. Não responde perguntas abertas, não
  conversa, não explica em prosa livre — apenas emite decisões estruturadas.
- NG3. Não tenta cobrir *todas* as classes de vulnerabilidade (OWASP Top 10
  completo) na v1. O MVP tem escopo fechado (seção 4.1).
- NG4. Não substitui revisão humana ou ferramentas de SAST estabelecidas —
  é uma camada adicional de triagem com confiança.
- NG5. Não processa ou envia código para serviços de terceiros na nuvem por
  padrão — inferência local ou em infraestrutura controlada pelo usuário.

## 4. Requirements

### 4.1. Escopo funcional do MVP

Categorias de vulnerabilidade cobertas na v1 (decidido em 2026-10-25):

| ID   | Categoria                          | Exemplos de padrão                                           |
|------|-------------------------------------|----------------------------------------------------------------|
| VC-1 | Injeção SQL / comando               | concatenação de string em query SQL, `os.system`/`subprocess` com input não sanitizado |
| VC-2 | Deserialização insegura             | `pickle.loads`, `yaml.load` sem `SafeLoader`, deserialização de objetos arbitrários |
| VC-3 | Segredos/credenciais expostas       | API keys, tokens, senhas hardcoded no código-fonte             |

Categorias fora do MVP, candidatas a v2 (registradas aqui para não se
perderem — inclui o exemplo original que motivou o projeto):

- Injeção de código dinâmico (`eval`, `exec`, `Function()`, `base64_decode`
  encadeado com execução).
- XSS / injeção em templates.
- Path traversal.
- Criptografia fraca/obsoleta.

### 4.2. Requisitos funcionais

- RF-1. Dado um `path` de arquivo ou diretório, o sistema lê o conteúdo via
  um repositório de código (abstração — local ou MCP).
- RF-2. O sistema constrói uma representação estrutural do código (AST no
  mínimo; grafo de fluxo de controle/dados como extensão) — ver ADR-0002.
- RF-3. O modelo de ML consome essa representação e emite, para cada
  categoria de VC-1 a VC-3, um score de confiança calibrado em [0, 1] e,
  quando aplicável, a localização exata do trecho suspeito.
- RF-4. O sistema serializa os achados em XML conforme schema versionado
  (a definir em `docs/adr/0004-xml-output-schema.md`).
- RF-5. O sistema não emite texto explicativo livre — apenas os campos
  estruturados do schema.
- RF-6. Quando rodando via MCP remoto (VPS), a tool de análise aceita um
  `path` relativo à raiz do projeto exposto pelo servidor, nunca um path
  arbitrário do sistema de arquivos do host.

### 4.3. Requisitos não-funcionais

- RNF-1. Arquitetura em camadas conforme Clean Architecture: `domain` não
  importa nada de `application`, `adapters` ou `infrastructure`;
  `application` depende apenas de `domain` e de interfaces (ports) que
  `infrastructure` implementa (inversão de dependência).
- RNF-2. Cobertura de testes unitários obrigatória para `domain` e
  `application` antes de qualquer merge na branch principal.
- RNF-3. Nomes de funções, classes e variáveis seguem as regras de Clean
  Code (nomes que revelam intenção, funções pequenas e com um único nível de
  abstração, sem efeitos colaterais escondidos).
- RNF-4. Latência de inferência local: meta inicial de referência (a validar
  empiricamente, não é compromisso rígido do MVP).
- RNF-5. O modelo deve poder ser re-treinado/re-calibrado sem alterar as
  camadas `application`/`adapters` (troca de modelo = troca de implementação
  do port `VulnerabilityModel`, não um redesign).

## 5. Architecture Decision Records

ADRs vivem em `docs/adr/`, numeradas sequencialmente, imutáveis após aceitas
(uma mudança de decisão gera uma ADR nova que supersede a anterior, nunca
edita a antiga). Registradas até agora:

- [ADR-0001](adr/0001-non-autoregressive-decision-architecture.md) — Arquitetura de decisão não-autoregressiva (não-transformer).
- [ADR-0002](adr/0002-graph-based-code-representation.md) — Representação do código como grafo (AST/CFG/DFG) via GNN.
- [ADR-0003](adr/0003-clean-architecture-layering.md) — Organização em camadas Clean Architecture.

- [ADR-0004](adr/0004-xml-output-schema.md) — Schema do XML de saída (v1).
- [ADR-0005](adr/0005-confidence-calibration-protocol.md) — Protocolo de
  calibração de confiança (temperature scaling por categoria, ECE).
- [ADR-0007](adr/0007-latent-iterative-reasoning.md) — Raciocínio latente
  iterativo no GSDM (múltiplos passos internos antes da decisão final, sem
  geração de linguagem).
- [ADR-0008](adr/0008-gsdm-0.1.0-implementation-baseline.md) — Base de
  implementação da GSDM 0.1.0 (Python como linguagem-alvo, NumPy puro,
  autograd próprio, limitações assumidas).

Pendente (marco 0.2.0):

- ADR-0006 — Transporte MCP: contrato da tool `analyze_path`, modo local vs.
  remoto. A 0.1.0 expõe apenas CLI.

## 6. Acceptance Criteria (para o MVP fechar)

- AC-1. Dado um arquivo de teste com uma injeção SQL conhecida (dataset de
  validação), o sistema retorna VC-1 com confiança > 0.8 e localização
  correta da linha.
- AC-2. Dado um arquivo "limpo" sem nenhum padrão das 3 categorias, o
  sistema não retorna nenhum achado com confiança > 0.5 (meta de baixo falso
  positivo, limiar a validar).
- AC-3. A saída XML valida contra o schema definido na ADR-0004 em 100% dos
  casos de teste.
- AC-4. A confiança reportada é calibrada: medindo Expected Calibration
  Error (ECE) no conjunto de validação, o erro fica abaixo de um limiar a
  definir na ADR-0005.
- AC-5. O mesmo caso de teste produz o mesmo resultado quando rodado local
  e quando rodado via MCP remoto (paridade de comportamento entre os dois
  modos de execução).
- AC-6. 100% das classes em `domain` e `application` têm teste unitário
  antes do merge na branch principal.

## 7. Glossário

- **GSDM (Graph-Structural Decision Model)**: a categoria/arquitetura de
  treinamento deste projeto — não-autoregressiva, opera sobre grafos
  estruturais (AST/CFG/DFG) em vez de texto solto, emite decisões tipadas
  com confiança calibrada (ver ADR-0001, ADR-0002, ADR-0007). É o nosso
  equivalente conceitual ao "System One Model" do Jev, mas nativo de
  estrutura de código, não de domínio genérico.
- **GVF (Graph-based Vulnerability Finder)**: o nome do modelo/produto
  específico deste projeto — uma instância do GSDM treinada para as
  categorias de vulnerabilidade do escopo (seção 4.1).
- **Confiança calibrada**: probabilidade que reflete a frequência real de
  acerto — diferente de um score arbitrário sem validação estatística.
- **Port**: interface definida na camada `application` que a camada
  `infrastructure` implementa (padrão de inversão de dependência da Clean
  Architecture).
- **MCP**: Model Context Protocol — protocolo usado para expor a tool de
  análise de código a um cliente, local ou remoto.
- **Raciocínio latente iterativo**: múltiplas rodadas de refinamento
  interno (message-passing sobre o grafo) antes da cabeça de decisão emitir
  o score final — equivalente estrutural de chain-of-thought, nunca
  verbalizado (ver ADR-0007).

## 8. Histórico de revisão

| Data       | Mudança                                                  |
|------------|-----------------------------------------------------------|
| 2026-10-25 | Rascunho inicial. Escopo MVP fechado em VC-1/VC-2/VC-3.   |
| 2026-10-25 | Projeto nomeado: arquitetura GSDM, modelo GVF. Licença definida (MPL 2.0). ADR-0007 (raciocínio latente) aceita. |
| 2026-10-25 | ADRs 0004, 0005 e 0008 aceitas; GSDM 0.1.0 implementada. MCP (ADR-0006) movido para 0.2.0. |
