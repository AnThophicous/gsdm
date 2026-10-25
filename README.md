# GVF — Graph-based Vulnerability Finder

Modelo de ML com arquitetura própria (GSDM — Graph-Structural Decision
Model: não-transformer, não-LLM generativo) para análise de
vulnerabilidades em código, com saída estruturada em XML e sistema de
confiança calibrado. Lê código localmente ou via MCP (local ou VPS remota).

**Licença:** [Mozilla Public License 2.0](LICENSE).

## Status — 0.1.0

A arquitetura GSDM e o pipeline do GVF estão implementados e testados:
autograd próprio, raciocínio latente iterativo, readout por atenção,
calibração por temperatura, parser Python → grafo, relatório XML e CLI.

**Ainda não existe um modelo GVF treinado.** O repositório não traz
dataset rotulado; a 0.1.0 entrega a arquitetura verificada, não um
detector pronto para uso real. Limitações assumidas em
[ADR-0008](docs/adr/0008-gsdm-0.1.0-implementation-baseline.md).

Ver [docs/SPEC.md](docs/SPEC.md) para o escopo completo e
[docs/adr/](docs/adr/) para as decisões de arquitetura.

## Uso

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```

Treinar a partir de um manifesto JSONL (um arquivo por linha, caminhos
relativos ao manifesto, rótulos `VC-1`/`VC-2`/`VC-3`; lista vazia = limpo):

```json
{"path": "samples/sql_concat.py", "labels": ["VC-1"]}
{"path": "samples/safe_query.py", "labels": []}
```

```bash
gvf train dataset/manifest.jsonl --output gvf.npz
gvf analyze path/to/project --model gvf.npz --threshold 0.5
```

`analyze` imprime o XML (schema v1, [ADR-0004](docs/adr/0004-xml-output-schema.md))
e sai com `0` (nada sinalizado), `1` (achados sinalizados) ou `2` (erro).

## Princípios do projeto

- **Clean Architecture** (Robert C. Martin) — regra de dependência
  unidirecional entre `domain → application → adapters → infrastructure`.
  Ver [ADR-0003](docs/adr/0003-clean-architecture-layering.md).
- **Clean Code** (Robert C. Martin) — nomes que revelam intenção, funções
  pequenas, um nível de abstração por função, sem efeitos colaterais
  escondidos. Aplicado em todo o código a partir do primeiro commit.
- **Pro Git** — ver seção "Fluxo de Git" abaixo.

## Fluxo de Git (convenções de Pro Git)

- `main` é sempre estável e deployável. Nunca commit direto nela além de
  merges revisados.
- Trabalho em branches de feature de vida curta: `feat/<nome>`,
  `fix/<nome>`, `docs/<nome>`, `refactor/<nome>`.
- Commits atômicos (uma mudança lógica por commit) no formato
  [Conventional Commits](https://www.conventionalcommits.org/):
  - `feat: adiciona parser de AST para Python`
  - `fix: corrige cálculo de confiança calibrada`
  - `docs: registra ADR-0004 sobre schema XML`
  - `refactor: extrai interface CodeRepository`
  - `test: cobre AnalyzeFileForVulnerabilities com casos VC-2`
- Mensagens de commit explicam o *porquê*, não só o *o quê* (o diff já
  mostra o quê).
- Nenhuma ADR aceita é editada depois do merge — uma mudança de decisão
  gera uma ADR nova que supersede a anterior.

## Estrutura (ver ADR-0003 e ADR-0008)

```
src/gsdm/              # A arquitetura de treinamento, genérica (não conhece Python nem vulnerabilidades)
  autograd/            # Diferenciação automática reverse-mode sobre NumPy
  nn/                  # Camadas, passo de raciocínio, encoder, cabeça de decisão, GSDModel
  graph.py             # Grafo estrutural heterogêneo
  optim.py             # Adam, clipping de gradiente
  calibration.py       # Temperature scaling, ECE
  training.py          # Trainer
  persistence.py       # .npz sem pickle
src/gvf/
  domain/              # Entidades e regras de negócio puras
  application/         # ports/ (interfaces) + use_cases/
  adapters/            # presenters/ (XML) + controllers/
  infrastructure/      # filesystem/, python_graph/ (parser → grafo), model/ (GSDM + treino)
  main.py              # Composition root (CLI)
tests/
  unit/                # gsdm e gvf, incluindo verificação numérica de gradientes
  integration/         # treino → análise → XML, ponta a ponta
  architecture/        # regra de dependência da Clean Architecture verificada por teste
docs/
  SPEC.md              # Especificação completa (Spec Kit)
  adr/                 # Architecture Decision Records
```

## Próximos passos

1. **Dataset rotulado em Python** para VC-1/VC-2/VC-3 — o bloqueio real
   para um GVF útil (os datasets públicos clássicos, como Big-Vul e
   Devign, são C/C++).
2. **0.2.0:** servidor MCP (ADR-0006) e treino em lote de vários grafos
   por vez (hoje o custo é dominado pelo overhead do Python por operação).
