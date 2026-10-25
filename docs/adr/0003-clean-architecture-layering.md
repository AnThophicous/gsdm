# ADR-0003: Organização em camadas Clean Architecture

- Status: Aceita
- Data: 2026-10-25

## Contexto

O requisito G5/RNF-1 exige que o projeto siga estritamente os princípios de
Clean Architecture (Robert C. Martin): regra de dependência unidirecional,
isolamento do domínio de frameworks e detalhes de infraestrutura, e
capacidade de substituir qualquer detalhe externo (modelo de ML, formato de
I/O, transporte) sem alterar as regras de negócio.

## Decisão

O código-fonte Python é organizado em quatro camadas, com a regra de
dependência: **camadas externas dependem de camadas internas; nunca o
contrário**. Dependências de camadas internas sobre comportamento externo
são feitas via interfaces (ports), implementadas pelas camadas externas
(inversão de dependência).

```
src/sentinel/
  domain/            # Entidades e regras de negócio puras.
                      # Sem import de application/adapters/infrastructure,
                      # sem import de bibliotecas de terceiros (exceto stdlib
                      # para tipos básicos).
                      # Ex: CodeUnit, VulnerabilityFinding, ConfidenceScore,
                      # VulnerabilityCategory.

  application/        # Casos de uso + ports (interfaces).
    ports/            # Interfaces que a infraestrutura implementa:
                      # CodeRepository, VulnerabilityModel, ResultPresenter.
    use_cases/        # Ex: AnalyzeFileForVulnerabilities.
                      # Depende apenas de domain + ports (nunca de
                      # implementações concretas).

  adapters/            # Interface adapters: traduzem entre use cases e
                       # o mundo externo.
    presenters/        # Ex: XmlFindingsPresenter (implementa ResultPresenter).
    controllers/        # Ex: McpAnalysisController — recebe a chamada MCP,
                        # aciona o use case, devolve a resposta formatada.

  infrastructure/      # Frameworks & drivers: detalhes concretos,
                       # substituíveis sem afetar camadas internas.
    filesystem/        # LocalFileCodeRepository (implementa CodeRepository).
    model/             # GnnVulnerabilityModel (implementa VulnerabilityModel).
    mcp/               # Servidor MCP (transporte), local ou remoto (VPS).
```

Regra de teste: um teste de `domain` ou `application` nunca importa nada de
`infrastructure`. Isso é verificável e será parte do critério de aceite
RNF-2/AC-6 — quebra de regra de dependência bloqueia merge.

## Alternativas consideradas

1. **Estrutura "flat" por tipo técnico (models/, utils/, services/)** —
   rejeitada: é o padrão mais comum em projetos Python, mas não impõe a
   regra de dependência; nada impede um "service" de importar direto do
   banco de dados ou do runtime do modelo, violando G5.
2. **Hexagonal Architecture (ports & adapters) sem as 4 camadas nomeadas
   de Clean Architecture** — é essencialmente compatível e foi nossa
   inspiração para nomear `application/ports`; optamos pelos nomes de
   Clean Architecture por serem os explicitamente pedidos no requisito G5.

## Consequências

- Positivo: trocar o motor de ML (ex: de GNN para outra arquitetura, se os
  experimentos indicarem) é uma troca de implementação em
  `infrastructure/model/`, sem tocar em `domain` ou `application` — atende
  diretamente RNF-5.
- Positivo: testável por construção — `domain`/`application` não precisam
  de mocks de I/O real, pois dependem de interfaces.
- Negativo: mais arquivos e indireção do que um script único — aceito como
  custo necessário dado o requisito explícito do usuário (projeto não é
  "descartável", precisa ser mantido e confiável a longo prazo).
- Pendência: convenção de testes (unit vs. integration, localização em
  `tests/`) será fixada antes do primeiro código de produção, junto da
  ADR-0004/0005/0006.
