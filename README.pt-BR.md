# GSDM — Graph-Structural Decision Model

**Uma arquitetura neural que não é transformer: raciocina sobre a estrutura do código-fonte e devolve decisões calibradas em vez de texto.**
Feita do zero em NumPy puro: autograd próprio, rede neural em grafo, raciocínio latente iterativo e calibração de confiança.
Inclui o **GVF (Graph-based Vulnerability Finder)**, o primeiro modelo construído sobre a GSDM: um detector de vulnerabilidades em Python, local e explicável, que responde em XML.

[English](README.md) · [Especificação](docs/SPEC.md) · [Decisões de arquitetura](docs/adr/)

---

## Por que a GSDM existe

LLMs geram texto um token por vez. Isso é a ferramenta errada quando a pergunta é
*"esse código é perigoso, quanta certeza você tem, e onde está o problema?"*. Não precisa de redação, precisa de decisão.

| | LLM transformer | GSDM |
|---|---|---|
| Entrada | sequência de tokens | **grafo**: árvore sintática, fluxo de statements, fluxo de dados definição→uso |
| Saída | texto gerado | **decisão tipada por categoria**: probabilidade + booleano + localização da evidência |
| "Pensamento" | chain-of-thought escrito em texto | **raciocínio latente iterativo**: N rodadas de message passing, nunca verbalizadas |
| Confiança | heurística sobre logits de token | **calibrada** por categoria (temperature scaling, medida com ECE) |
| Execução | GPU pesada, API na nuvem | **NumPy num notebook**, 100% local |

## Como funciona

1. **Grafo estrutural.** Cada nó do AST vira um nó do grafo. Arestas ligam pai e filho, statements consecutivos e a
   definição de cada nome aos seus usos (por exemplo `import pickle` → `pickle.loads(...)`).
2. **Raciocínio latente iterativo.** Um único passo de raciocínio, com pesos compartilhados, roda N vezes. A cada
   rodada a informação anda mais um passo no fluxo de dados. Como os pesos são compartilhados, N pode mudar na
   inferência: pensar mais em código difícil.
3. **Readout por atenção por categoria.** O nó com maior atenção é reportado como **localização da evidência**,
   sem precisar de rótulo por linha no treino.
4. **Calibração.** Temperature scaling por categoria, ajustado num split de validação separado.

## GVF

| Código | Categoria |
|---|---|
| VC-1 | Injeção SQL / de comando |
| VC-2 | Deserialização insegura |
| VC-3 | Segredos hardcoded |

São **sinais aprendidos, não regras de regex**. A saída é XML sem texto livre
([schema v1](docs/adr/0004-xml-output-schema.md)), com código de saída `0` (nada sinalizado), `1` (achados) ou `2` (erro).

## Uso rápido

```bash
git clone https://github.com/AnThophicous/gsdm.git
cd gsdm
python -m venv .venv
.venv/bin/pip install -e ".[dev]"      # Windows: .venv\Scripts\pip
.venv/bin/pytest
```

Manifesto de treino em JSONL (caminhos relativos ao manifesto; lista vazia = código limpo):

```json
{"path": "samples/sql_concat.py", "labels": ["VC-1"]}
{"path": "samples/safe_query.py", "labels": []}
```

```bash
gvf train dataset/manifest.jsonl --output gvf.npz
gvf analyze caminho/do/projeto --model gvf.npz --threshold 0.5
```

## Status (1.0.0): leia antes de usar

- ✅ Arquitetura, treino, calibração, parser Python → grafo, relatório XML e CLI implementados e testados.
- ✅ Cada operação do autograd e o modelo completo têm o gradiente **verificado contra diferenças finitas**.
- ✅ O modelo aprende e generaliza tarefas relacionais em grafo que não se resolvem sem raciocinar sobre a estrutura.
- ⚠️ **Nenhum modelo GVF treinado vem com o repositório.** Ainda não existe aqui um dataset rotulado de
  vulnerabilidades em Python. Um modelo treinado num conjunto sintético pequeno decora os exemplos e erra código
  novo, então **não** use o GVF para decisões de segurança antes de treiná-lo com dados reais.
- Limitações desta versão: fluxo de controle simplificado, def-use por posição no fonte, um arquivo por vez.
  Ver [ADR-0008](docs/adr/0008-gsdm-0.1.0-implementation-baseline.md).

## Roadmap

- **Dataset** rotulado em Python para VC-1, VC-2 e VC-3.
- **1.1.0:** servidor MCP para editores e agentes de IA chamarem o GVF localmente ou numa VPS, e treino em lote.
- Depois: profundidade de raciocínio adaptativa, mais linguagens, fluxo de dados entre arquivos.

## Princípios de engenharia

- **Clean Architecture**, com a regra de dependência **verificada por teste** ([ADR-0003](docs/adr/0003-clean-architecture-layering.md)).
- **Clean Code** e **spec-driven**: toda decisão vira [ADR](docs/adr/) antes do código.
- **Pro Git**: commits atômicos no formato Conventional Commits, branches de feature de vida curta, `main` só recebe
  merges revisados, e ADRs aceitas nunca são editadas (uma nova ADR substitui a anterior).
- **Seguro por construção**: modelos são `.npz` carregados com `allow_pickle=False`.

## Licença

[Mozilla Public License 2.0](LICENSE). Pode ser usada em produtos de código fechado; mudanças nos arquivos da
própria GSDM precisam ser compartilhadas sob a mesma licença.
