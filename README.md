# Sentinel (nome provisório)

Motor de ML com arquitetura própria (não-transformer, não-LLM generativo)
para análise de vulnerabilidades em código, com saída estruturada em XML e
sistema de confiança calibrado. Lê código localmente ou via MCP (local ou
VPS remota).

## Status

**Em fase de especificação.** Nenhum código de produção foi escrito ainda.
Ver [docs/SPEC.md](docs/SPEC.md) para o escopo completo e
[docs/adr/](docs/adr/) para as decisões de arquitetura já tomadas.

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

## Estrutura (planejada, ver ADR-0003)

```
src/sentinel/
  domain/            # Entidades e regras de negócio puras
  application/       # Casos de uso + ports (interfaces)
  adapters/          # Presenters e controllers
  infrastructure/     # Filesystem, modelo de ML, servidor MCP
docs/
  SPEC.md            # Especificação completa (Spec Kit)
  adr/               # Architecture Decision Records
tests/
  unit/
  integration/
```

## Próximos passos

Ver seção 5 de [docs/SPEC.md](docs/SPEC.md) — ADRs pendentes antes do
primeiro código de produção: schema XML (0004), protocolo de calibração de
confiança (0005), contrato da tool MCP (0006).
