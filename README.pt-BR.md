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

Manifesto de treino em JSONL. Caminhos relativos ao manifesto, lista vazia = código limpo, e `group` diz de
onde o arquivo veio (projeto, CVE, template):

```json
{"path": "samples/sql_concat.py", "labels": ["VC-1"], "group": "cve-2024-1234"}
{"path": "samples/safe_query.py", "labels": [], "group": "api-interna"}
```

```bash
gvf capacity --hidden-size 128                        # parâmetros e memória antes de treinar
gvf train dataset/manifest.jsonl --output gvf.npz     # logs, painel e checkpoint ficam em gvf.run/
gvf train dataset/manifest.jsonl --output gvf.npz --epochs 60 --resume   # retoma um treino interrompido
gvf analyze caminho/do/projeto --model gvf.npz --threshold 0.5
```

Para testar sem instalar nada: [notebook no Google Colab](https://colab.research.google.com/github/AnThophicous/gsdm/blob/main/notebooks/gsdm_colab.ipynb).

## Está aprendendo ou decorando?

- **Validação por grupo:** um grupo inteiro vai só para o treino ou só para a validação, então a validação mede
  código de origens que o modelo nunca viu.
- **Logs de treino no estilo de LLM**, uma linha por época (loss de treino e validação, gap, erro de calibração,
  F1 por categoria, grafos/s), também gravados em `metrics.jsonl`.
- **Painel ao vivo** (`dashboard.html`, um arquivo só, sem scripts) com as curvas de treino e validação.
- **Diagnóstico automático:** `learning`, `memorizing`, `stalled` ou `no-validation`, com a evidência. No dataset
  sintético de demonstração ele responde, por exemplo: *validation loss rose 28% since epoch 9 while training
  loss fell 84%*.
- **Early stopping** com exportação dos melhores pesos de validação, e **checkpoints** que retomam o treino de
  forma bit a bit idêntica (verificado por teste).

## Desempenho

Medido com `python benchmarks/benchmark.py` na máquina de referência: Celeron N4100 (4 núcleos, sem AVX2),
3,8 GB de RAM, sem GPU. Mesmo modelo de 53.254 parâmetros nas duas versões.

| | 1.0.0 | 1.1.0 |
|---|---|---|
| Treino | 39 grafos/s | **116 grafos/s** (3×) |
| Analisar um arquivo de 2.000 linhas | 868 ms | **637 ms** |
| Pico de memória na análise | 34,5 MB | **15,1 MB** |

A GSDM 1.1.0 treina em um processo, em CPU. `gvf capacity --target-parameters 1e10` mostra que 10 bilhões de
parâmetros exigiriam cerca de 397 GB de memória de treino: isso é trabalho para várias GPUs com treino
distribuído, que esta versão não oferece.

## Status (1.1.0): leia antes de usar

- ✅ Arquitetura, treino em lote, calibração, parser Python → grafo, relatório XML e CLI implementados e testados.
- ✅ Cada operação do autograd e o modelo completo têm o gradiente **verificado contra diferenças finitas**.
- ✅ O modelo aprende e generaliza tarefas relacionais em grafo que não se resolvem sem raciocinar sobre a estrutura.
- ✅ O treino informa se o modelo está aprendendo ou decorando.
- ⚠️ **Nenhum modelo GVF treinado vem com o repositório.** Ainda não existe aqui um dataset rotulado de
  vulnerabilidades em Python. Um modelo treinado num conjunto sintético pequeno decora os exemplos e erra código
  novo, então **não** use o GVF para decisões de segurança antes de treiná-lo com dados reais.
- Limitações desta versão: fluxo de controle simplificado, def-use por posição no fonte, um arquivo por vez.
  Ver [ADR-0008](docs/adr/0008-gsdm-0.1.0-implementation-baseline.md).

## Roadmap

- **Dataset** (em andamento, [ADR-0010](docs/adr/0010-cvefixes-dataset.md)): correções reais de CVEs do
  [CVEfixes](https://zenodo.org/records/13118970), em pares antes/depois e agrupadas por repositório. O primeiro
  passo, o censo de quantos exemplos em Python existem, roda com `python -m dataset_tools.cvefixes` (seção 8 do
  notebook do Colab).
- **Backend de GPU** opcional (CuPy, placas NVIDIA), verificado contra os resultados da CPU no Colab.
- **Servidor MCP** para editores e agentes de IA chamarem o GVF localmente ou numa VPS.
- Depois: features esparsas, profundidade de raciocínio adaptativa, mais linguagens, fluxo de dados entre arquivos.

O que mudou em cada versão está no [CHANGELOG.md](CHANGELOG.md).

## Princípios de engenharia

- **Clean Architecture**, com a regra de dependência **verificada por teste** ([ADR-0003](docs/adr/0003-clean-architecture-layering.md)).
- **Clean Code** e **spec-driven**: toda decisão vira [ADR](docs/adr/) antes do código.
- **Pro Git**: commits atômicos no formato Conventional Commits, branches de feature de vida curta, `main` só recebe
  merges revisados, e ADRs aceitas nunca são editadas (uma nova ADR substitui a anterior).
- **Seguro por construção**: modelos são `.npz` carregados com `allow_pickle=False`.

## Licença

[Mozilla Public License 2.0](LICENSE). Pode ser usada em produtos de código fechado; mudanças nos arquivos da
própria GSDM precisam ser compartilhadas sob a mesma licença.
