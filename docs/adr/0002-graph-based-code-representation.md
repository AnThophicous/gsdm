# ADR-0002: Representação do código como grafo (AST/CFG/DFG) via GNN

- Status: Aceita
- Data: 2026-10-25

## Contexto

Padrões de vulnerabilidade das categorias do MVP (VC-1 injeção SQL/comando,
VC-2 deserialização insegura, VC-3 segredos expostos) são majoritariamente
**estruturais e de fluxo de dados**: o que importa não é só "essa string
aparece no arquivo", mas "esse dado não-confiável chega, sem sanitização,
a esse sink perigoso" (ex: input de usuário chegando a uma query SQL sem
parametrização).

Tratar o código como sequência de tokens (abordagem tipo transformer de
texto) perde essa informação estrutural ou exige que o modelo a infira
implicitamente a partir de grandes volumes de dados — caro e menos
confiável para o nosso orçamento de treino.

## Decisão

O código de entrada é parseado para:

1. **AST** (Abstract Syntax Tree) — estrutura sintática, obrigatório.
2. **CFG** (Control Flow Graph) — ordem de execução possível, usado para
   localizar sinks perigosos alcançáveis.
3. **DFG** (Data Flow Graph) — rastreamento de de-para de dados não
   confiáveis (ex: parâmetro de função HTTP) até um sink (ex: `cursor.execute`).

Esses três grafos são combinados em um único grafo heterogêneo por unidade
de código analisada, que alimenta uma **GNN (Graph Neural Network)** como
encoder. A cabeça de decisão (ADR-0001) consome o embedding agregado do
grafo e emite os scores por categoria.

Esta é a mesma linha de abordagem usada em pesquisa acadêmica de detecção
de vulnerabilidade (ex: Devign, VulDeePecker), adaptada e implementada do
zero para este projeto — não é uma dependência de framework de terceiros
para a arquitetura em si (bibliotecas de parsing de AST da própria
linguagem-alvo podem ser usadas; a GNN é implementada no projeto).

## Alternativas consideradas

1. **Tokens brutos + encoder sequencial (LSTM/CNN 1D)** — mais simples de
   implementar, mas perde a relação estrutural explícita entre
   fonte-de-dado e sink; fica como baseline de comparação nos experimentos,
   não como arquitetura principal.
2. **Apenas AST sem CFG/DFG** — mais simples, mas insuficiente para VC-1
   (injeção), que depende fundamentalmente de fluxo de dados; rejeitado
   como único sinal.

## Consequências

- Positivo: representação alinhada com a natureza real do problema
  (fluxo de dados até um sink), reduz a quantidade de dados de treino
  necessária comparado a aprender isso puramente de texto.
- Negativo: parsing de CFG/DFG é mais complexo de implementar e manter do
  que tokenização simples; precisa de um parser por linguagem suportada.
- Risco: precisão do CFG/DFG depende da qualidade do parser — bugs no
  parser geram ruído silencioso no treino. Mitigação: testes unitários
  dedicados ao parser (RNF-2) com casos conhecidos antes de treinar
  qualquer modelo em cima dele.
- Decisão em aberto para depois: linguagem(ns) de código-alvo suportadas no
  MVP (a definir antes de iniciar a implementação do parser).
