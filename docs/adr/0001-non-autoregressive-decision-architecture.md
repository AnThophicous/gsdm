# ADR-0001: Arquitetura de decisão não-autoregressiva (não-transformer)

- Status: Aceita
- Data: 2026-10-25

## Contexto

O objetivo do sistema é classificar trechos de código em categorias de
vulnerabilidade com um grau de confiança. Isso é, por natureza, um problema
de **decisão estruturada** (classificação/scoring), não um problema de
**geração de linguagem**.

Modelos transformer autoregressivos (LLMs tradicionais, GPT-like) são
otimizados para gerar sequências de texto token a token. Usar essa família
de arquitetura para um problema de classificação é possível, mas:

- desperdiça capacidade computacional gerando texto quando só precisamos de
  um número por categoria;
- dificulta calibração de confiança (a "confiança" de um LLM generativo
  normalmente vem de heurísticas sobre logits de token, não é um score
  desenhado para isso);
- é mais caro e mais lento do que necessário para a tarefa.

A TypeSafe AI (produto Jev) validou publicamente que uma arquitetura
não-autoregressiva, com saída tipada e probabilidade calibrada por decisão,
é viável e drasticamente mais rápida/barata para esse tipo de tarefa
(70-500ms, segundo material da própria empresa).

## Decisão

O modelo do projeto Sentinel será **não-autoregressivo**: dado um código de
entrada (representado estruturalmente, ver ADR-0002), o modelo emite, em um
único passe, um vetor de scores — um por categoria de vulnerabilidade
definida no escopo (VC-1, VC-2, VC-3) — mais a localização do achado quando
aplicável.

Isso é uma escolha de **categoria de arquitetura**, não uma cópia do Jev: a
arquitetura interna (camadas, representação de entrada, método de treino)
é desenhada do zero neste projeto (ver ADR-0002).

## Alternativas consideradas

1. **Fine-tuning de um LLM existente para classificação** — rejeitado:
   contradiz o requisito G1/G5 de arquitetura própria, e traz todo o custo
   computacional de um LLM generativo para um problema que não precisa de
   geração.
2. **Transformer encoder-only (tipo BERT) treinado do zero** — considerado
   viável, mas rejeitado como escolha principal porque a atenção densa é
   cara para representações estruturais grandes (árvores/grafos de código);
   fica registrado como alternativa de fallback se a abordagem em grafo
   (ADR-0002) não atingir os critérios de aceite.
3. **Regras fixas / regex (SAST clássico)** — rejeitado: não escala para
   "entender contexto do projeto" (requisito central do Problem Statement),
   alto índice de falso positivo, não aprende.

## Consequências

- Positivo: inferência rápida e barata; confiança calibrada é um objetivo de
  primeira classe no treino, não um hack pós-hoc; saída naturalmente tipada
  facilita o schema XML (RF-4/RF-5).
- Negativo: menos "flexível" que um LLM — se o escopo crescer para precisar
  de explicações em linguagem natural, será necessário um componente
  adicional (fora do escopo deste projeto, ver Non-Goals NG2).
- Risco: calibração de confiança é difícil de acertar sem dataset de
  validação robusto — mitigação detalhada na ADR-0005 (pendente).
