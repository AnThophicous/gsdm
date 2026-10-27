# Especificação do dataset do GVF (para agentes de coleta)

Este documento é a tarefa completa para quem vai coletar e montar o dataset de treino do GVF, o modelo de
detecção de vulnerabilidades construído sobre a GSDM. Ele foi escrito para um agente de IA (por exemplo, o
Codex) executar sem precisar adivinhar nada. Onde houver dúvida, **a regra é excluir o exemplo, não chutar o
rótulo**: um rótulo errado ensina o modelo errado, um exemplo a menos não.

Palavras como **DEVE**, **NÃO DEVE** e **PODE** têm o sentido de obrigação, proibição e opção.

Base desta especificação: [ADR-0009](adr/0009-efficient-cpu-training-and-learning-diagnostics.md) (validação
por grupo) e [ADR-0010](adr/0010-cvefixes-dataset.md) (CVEfixes como fonte inicial).

---

## 1. Leia antes de começar

1. `docs/adr/0009-efficient-cpu-training-and-learning-diagnostics.md` e `docs/adr/0010-cvefixes-dataset.md`.
2. `src/gvf/infrastructure/model/training_pipeline.py`, funções `_parse_manifest_line` e `_to_examples`:
   é exatamente assim que o manifesto é lido.
3. `dataset_tools/cvefixes/schema.py`: a tabela `CWE_CATEGORIES` é a **única fonte** do mapeamento de CWE
   para categoria. Importe-a; NÃO DEVE copiar a tabela para outro lugar.
4. `dataset_tools/cvefixes/`: ferramentas já prontas para o CVEfixes (`prepare` e `census`).

## 2. Como a GSDM enxerga um exemplo

- Um exemplo é **um arquivo Python inteiro**. Ele vira um grafo (árvore sintática, sequência de statements e
  fluxo definição→uso de nomes), e o modelo toma **três decisões independentes** sobre ele: tem ou não tem
  VC-1, VC-2 e VC-3.
- O rótulo é **por arquivo e multi-rótulo**: um arquivo pode ter zero, uma ou várias categorias.
- O modelo não lê texto livre, comentários de commit nem metadados: só o código do arquivo.
- A validação é feita pela própria GSDM, separando **grupos inteiros** entre treino e validação. Por isso o
  campo `group` (seção 7) é tão importante quanto o rótulo.

## 3. O que entregar

```
gvf-dataset/
  manifest.jsonl            treino + validação (a GSDM separa a validação sozinha, por grupo)
  test-manifest.jsonl       teste final, grupos que nunca aparecem no manifest.jsonl (seção 11)
  shard-00001.jsonl         o mesmo conteúdo do manifest.jsonl dividido em partes carregáveis (seção 12),
  shard-00002.jsonl         na raiz, para que os caminhos sejam os mesmos do manifest.jsonl
  ...
  files/<grupo-normalizado>/<sha256-16>.py
  ATTRIBUTION.md            fontes, licenças e citações exigidas
  README.md                 datasheet do dataset (seção 14)
  stats.json                contagens geradas por script, nunca à mão
  audit/sample.jsonl        amostra para revisão humana (seção 13)
```

- O dataset **NÃO DEVE** ser commitado no repositório: o código vem de milhares de projetos com licenças
  próprias. O repositório recebe só o código das ferramentas de coleta (seção 17).
- Os nomes de arquivo DEVEM ser o hash do conteúdo (`sha256` dos primeiros 16 caracteres hexadecimais), não o
  nome original, para evitar colisões. O nome e o caminho originais vão em `meta`.

## 4. Formato do manifesto (contrato exato)

Um objeto JSON por linha, UTF-8 **sem BOM** (um BOM na primeira linha quebra a leitura). Linhas em branco são
ignoradas.

```json
{"path": "files/github.com__psf__requests/3f2a9c1e8b7d6a54.py", "labels": ["VC-1"], "group": "github.com/psf/requests", "meta": {"source": "cvefixes", "role": "vulnerable_before_fix", "pair_id": "cvefixes:CVE-2023-1234:app/db.py", "cve": ["CVE-2023-1234"], "cwe": ["CWE-89"], "repo_url": "https://github.com/psf/requests", "commit": "9f1c...", "original_path": "app/db.py", "license": "Apache-2.0", "sha256": "3f2a9c1e...", "lines": 120, "nodes": 431, "evidence_lines": [12, 13], "secret_redacted": false, "collected_at": "2026-10-27"}}
```

Campos que a GSDM lê:

| campo | obrigatório | regra |
|---|---|---|
| `path` | sim | relativo à pasta do manifesto, com `/`, sem `..`, sem caminho absoluto; o arquivo DEVE existir |
| `labels` | sim | lista com zero ou mais de `"VC-1"`, `"VC-2"`, `"VC-3"`; lista vazia = arquivo limpo. **Qualquer outro valor aborta o treino inteiro** |
| `group` | sim nesta especificação | string; regras na seção 7 |

O campo `meta` é ignorado pela GSDM e DEVE ser usado para toda a procedência. Valores de `meta.role`:

| role | o que é |
|---|---|
| `vulnerable_before_fix` | arquivo antes de uma correção real de vulnerabilidade |
| `fixed_after_fix` | o mesmo arquivo depois da correção (negativo difícil para a categoria corrigida) |
| `clean_same_repo` | outro arquivo do mesmo repositório, sem vulnerabilidade conhecida |
| `safe_api_usage` | código que usa as mesmas APIs de forma segura (seção 6) |
| `synthetic` | código gerado por template ou por modelo de linguagem (seção 10) |

Outras CWEs fora do escopo (por exemplo CWE-79 ou CWE-94) PODEM ser registradas em `meta.cwe` para categorias
futuras, mas **NUNCA** em `labels`.

## 5. Definição operacional das categorias

O mapeamento de CWE é o de `CWE_CATEGORIES` (CWE-89, 77, 78, 88 → VC-1; CWE-502 → VC-2; CWE-798, 259, 321 →
VC-3). Para exemplos que não vêm de um CVE com CWE, valem as definições abaixo. O arquivo é positivo se
contém **pelo menos uma** ocorrência.

**VC-1 — injeção SQL ou de comando**
- Positivo: dado não constante chega a uma execução SQL por string montada (`+`, `%`, `.format`, f-string) em
  `execute`, `executemany`, `raw`, `text(...)`; ou a execução de comando com string montada em `os.system`,
  `os.popen`, `subprocess.*(..., shell=True)`, `commands.getoutput`.
- Negativo: consulta parametrizada (`execute("... %s", (x,))`), ORM sem SQL cru, `subprocess.run([...])` com
  lista de argumentos e sem `shell=True`, comandos totalmente constantes.

**VC-2 — deserialização insegura**
- Positivo: `pickle`/`cPickle`/`_pickle`/`dill`/`shelve`/`marshal` `.load(s)` sobre dados que não são
  comprovadamente locais e confiáveis; `yaml.load` sem `SafeLoader`, `yaml.unsafe_load`; `jsonpickle.decode`;
  `torch.load(..., weights_only=False)` sobre arquivo externo.
- Negativo: `json.loads`, `yaml.safe_load`, `yaml.load(..., Loader=yaml.SafeLoader)`, `pickle.dumps` (gravar não
  é o problema).

**VC-3 — segredo no código**
- Positivo: senha, chave de API, token, chave privada ou string de conexão com senha **escrita literalmente**
  no código e usada como credencial.
- Negativo: placeholders (`""`, `"changeme"`, `"<YOUR_API_KEY>"`, `"xxx"`, `"test"`, `"password"` em teste
  óbvio), leitura de `os.environ` / `os.getenv` / arquivo de configuração / gerenciador de segredos.
- Arquivos de teste e exemplos com credenciais claramente falsas NÃO DEVEM ser positivos. Se não der para
  decidir se é real ou placeholder, **exclua o arquivo**.

## 6. Composição

Metas mínimas para um treino que diga algo (o censo do CVEfixes vai mostrar o quanto dá para alcançar):

| item | mínimo | recomendado |
|---|---|---|
| positivos por categoria | 300 | 2.000+ |
| grupos (repositórios) distintos com positivos, por categoria | 50 | 300+ |
| negativos no total | 3× o número de positivos | 4× |
| `safe_api_usage` por categoria | 1× os positivos da categoria | 2× |

- Os `safe_api_usage` são **obrigatórios**. Sem eles o modelo aprende atalhos como "tem `import pickle` = é
  vulnerável" ou "tem `execute` = é injeção", e erra em todo código seguro que usa as mesmas bibliotecas.
- Todo `vulnerable_before_fix` DEVE vir acompanhado do seu `fixed_after_fix` quando o arquivo depois da
  correção existe e passa nas validações.
- O `fixed_after_fix` recebe os rótulos que ele **ainda** tem. Se a correção tirou só a injeção mas sobrou um
  segredo no arquivo, ele é `["VC-3"]`, não `[]`.

## 7. Grupos (o que impede o modelo de "colar")

- `group` = identidade canônica do **repositório**: `host/dono/repo` em minúsculas, sem `.git`, sem barra no
  fim (`github.com/psf/requests`).
- Forks DEVEM usar o grupo do repositório original quando ele puder ser identificado.
- Todos os arquivos de um mesmo `pair_id` DEVEM ter o mesmo `group`.
- Exemplos sintéticos: um grupo por template ou por prompt de geração (`synthetic/template-<id>`), nunca um
  grupo por arquivo.
- Um mesmo conteúdo em dois grupos diferentes é vazamento: resolva pela deduplicação (seção 8).

## 8. Deduplicação

1. **Exata**: `sha256` do conteúdo normalizado (finais de linha `\n`, espaços no fim de linha removidos). Um
   hash aparece **uma vez** no dataset inteiro (manifest + teste). O primeiro visto fica; os outros são
   descartados e contados em `stats.json`.
2. **Quase duplicada**: arquivos com similaridade de tokens ≥ 0,9 (por exemplo MinHash sobre shingles de 5
   tokens) em **grupos diferentes** DEVEM ir para o mesmo grupo ou ter um dos dois descartado. É o caso típico
   de código vendorizado ou copiado entre projetos.

## 9. Restrições de arquivo

Cada arquivo DEVE passar em todas, e a falha de qualquer uma exclui o arquivo:

- Faz parse com `ast.parse` no **Python 3.12** (a menor versão suportada). Sintaxe nova demais (por exemplo
  template strings do 3.14) e Python 2 ficam de fora. **Um único arquivo que não faz parse aborta o treino
  inteiro**, então isso DEVE ser verificado antes da entrega.
- Não contém byte nulo e é UTF-8 válido (transcodifique de outra codificação ou exclua).
- Não estoura recursão nem memória no parse (`RecursionError`, `MemoryError`).
- No máximo **5.000 linhas** e **20.000 nós** de grafo (`PythonGraphBuilder().build(...).structure.node_count`).
- Não é código gerado, minificado ou vendorizado: excluir `site-packages/`, `vendor/`, `third_party/`,
  `node_modules/`, migrações geradas automaticamente, arquivos `_pb2.py` e similares.

## 10. Fontes, legalidade e segurança

Ordem de prioridade:

1. **CVEfixes 1.0.8** (CC BY 4.0) com as ferramentas de `dataset_tools/cvefixes/`.
2. **Avisos de segurança de pacotes PyPI** (GitHub Advisory Database / OSV) que apontam o commit da correção:
   mesmo esquema antes/depois.
3. **Repositórios públicos com licença permissiva** (MIT, BSD, Apache-2.0) para `clean_same_repo` e
   `safe_api_usage`.
4. **Sintéticos**, por último e com limite (abaixo).

Regras obrigatórias:

- Registrar a licença de cada arquivo em `meta.license`. Licença desconhecida PODE entrar, mas é contada à
  parte em `stats.json`.
- Respeitar termos de uso e limites de taxa das APIs (GitHub, OSV, NVD). NÃO DEVE coletar de repositórios
  privados nem contornar autenticação.
- **NUNCA executar o código coletado** nem instalar pacotes coletados. O código é tratado só como texto.
  Projetos vulneráveis e maliciosos fazem parte do material.
- **Segredos reais NÃO DEVEM ser redistribuídos.** Em todo arquivo VC-3, e em qualquer arquivo onde apareça um
  segredo que pareça real, o valor literal DEVE ser trocado por um valor aleatório de mesmo tamanho e mesmas
  classes de caracteres, mantendo prefixos de formato conhecidos (`AKIA`, `ghp_`, `sk_live_`, `xoxb-`, cabeçalho
  `-----BEGIN ... PRIVATE KEY-----`), porque o formato é o que o modelo deve aprender. Marcar
  `meta.secret_redacted: true`. NUNCA testar se uma credencial encontrada funciona.
- **Sintéticos** (template, mutação ou gerado por modelo de linguagem): `meta.role = "synthetic"`, no máximo
  **10% dos positivos** de cada categoria, e **nunca** no `test-manifest.jsonl`. A versão 1.1.0 mostrou que o
  modelo decora templates; dado sintético é tempero, não base.

## 11. Conjunto de teste

- A validação dentro do `manifest.jsonl` é usada pela GSDM para early stopping e calibração, então ela deixa de
  ser uma medida imparcial. O teste final precisa ser separado.
- `test-manifest.jsonl` recebe os grupos cujo `sha256(group)` em hexadecimal termina em `0` (cerca de 1 em 16
  grupos), determinístico e sem escolha manual.
- Nenhum grupo, `pair_id` ou hash de conteúdo do teste PODE aparecer no `manifest.jsonl`.
- Se houver datas de CVE, registrar em `stats.json` o intervalo de datas do treino e do teste.

## 12. Tamanho e memória (leia antes de coletar "um dataset gigantesco")

A GSDM 1.1.0 carrega **todos os grafos de um manifesto na RAM de uma vez** antes de treinar. Medido em
arquivos Python reais: cerca de **1,7 KB por nó** e **3,5 nós por linha** (uns 3,5 MB por arquivo médio).

| máquina | RAM | teto aproximado por manifesto |
|---|---|---|
| máquina de referência (Celeron N4100) | 3,8 GB | 1 milhão de nós |
| Google Colab, RAM alta | ~50 GB | 25 milhões de nós |

Por isso:

- Calcule `meta.nodes` de cada arquivo e o total por manifesto.
- Além do `manifest.jsonl` completo, entregue `shard-NNNNN.jsonl` na raiz do dataset, com **no máximo 5
  milhões de nós cada**, com **grupos inteiros** (um grupo nunca é dividido entre shards) e proporções de
  categoria parecidas entre os shards. As linhas dos shards são cópias exatas das linhas do `manifest.jsonl`.
- Treinar com o dataset inteiro de uma vez, acima desse teto, exige um carregador por streaming na GSDM, que
  **ainda não existe**. Um dataset maior que o teto não é erro: ele é treinado por shards até o carregador
  existir.

## 13. Auditoria de qualidade

- Gerar `audit/sample.jsonl` com 200 entradas sorteadas com seed fixa, estratificadas por categoria e por
  `role`, para revisão humana.
- Registrar no `README.md` do dataset a taxa de erro encontrada na revisão, quando ela for feita.

## 14. Datasheet (`README.md` do dataset)

Conteúdo mínimo: data da coleta e versões de cada fonte; como cada rótulo foi obtido; contagens por categoria,
`role`, fonte e licença (copiadas do `stats.json`); descartes por motivo; ruído de rótulo conhecido (o "depois"
de uma correção pode ter outras falhas); como reproduzir a coleta (comandos e seeds); a atribuição exigida pela
CC BY 4.0 do CVEfixes.

## 15. Checklist de aceite

Todos os itens DEVEM passar, verificados por script, antes de entregar:

1. Todas as linhas dos manifestos são JSON válido, sem BOM, com `path`, `labels` e `group`.
2. `labels` só contém `VC-1`, `VC-2`, `VC-3`.
3. Todo `path` é relativo, sem `..`, e o arquivo existe.
4. Todo arquivo passa nas restrições da seção 9, verificadas no Python 3.12.
5. Nenhum hash de conteúdo se repete no dataset inteiro.
6. Todo `pair_id` tem um único `group`.
7. Nenhum grupo, `pair_id` ou hash do teste aparece no treino.
8. Todo arquivo com `VC-3` tem `meta.secret_redacted: true`.
9. Sintéticos ≤ 10% dos positivos de cada categoria e nenhum no teste.
10. Cada shard tem ≤ 5 milhões de nós e nenhum grupo dividido.
11. `stats.json` bate com uma recontagem feita a partir dos manifestos.
12. Teste de fumaça com a CLI real, em um shard:

    ```bash
    gvf train gvf-dataset/shard-00001.jsonl --output /tmp/smoke.npz --epochs 1 --hidden-size 8 --reasoning-steps 1
    ```

    Deve terminar com código 0, **sem** o aviso `validation is not grouped` e sem aviso de categoria sem
    positivo na validação.

## 16. Erros proibidos (os mais prováveis)

- Rotular com grep, regex ou com a saída de uma ferramenta SAST como o Bandit. O modelo aprenderia a imitar a
  ferramenta, inclusive os falsos positivos.
- Colocar o antes e o depois da mesma correção em grupos diferentes.
- Fazer a divisão de treino e validação por conta própria, ou de forma aleatória por arquivo.
- Pôr código de categoria fora do escopo (XSS, path traversal, `eval`) em `labels`.
- Deixar passar um arquivo que não faz parse no Python 3.12.
- Redistribuir um segredo real.
- Executar qualquer código coletado.
- Usar o nome original do arquivo como nome no dataset.
- Gerar `stats.json` à mão.

## 17. Se for escrever código de coleta neste repositório

- Coletores novos em `dataset_tools/<fonte>/`, seguindo o padrão de `dataset_tools/cvefixes/`: aviso da MPL 2.0
  no topo de cada arquivo, funções pequenas com nomes que revelam a intenção, sem efeitos colaterais escondidos.
- Testes em `tests/dataset_tools/`, contra dados falsos, **sem acesso à rede**.
- `python -m pytest` DEVE continuar passando.
- Commits atômicos no formato Conventional Commits, numa branch `feat/<nome>`, nunca direto na `main`. Autor:
  `Brenno Leon <273256966+AnThophicous@users.noreply.github.com>`, **sem** linha `Co-Authored-By` de assistente
  de IA.
- Nenhum arquivo do dataset, nenhuma credencial e nenhum dump do CVEfixes entra no git.
