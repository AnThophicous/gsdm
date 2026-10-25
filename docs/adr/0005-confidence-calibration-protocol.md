# ADR-0005: Protocolo de calibração de confiança

- Status: Aceita
- Data: 2026-10-25

## Contexto

G3 exige confiança calibrada: "0.8" deve significar ~80% de acerto
empírico. Redes neurais treinadas com cross-entropy tendem a ficar
superconfiantes, então o score cru (sigmoid do logit) não serve como
confiança sem uma etapa de calibração explícita.

## Decisão

1. **Método:** temperature scaling **por categoria**. Para cada categoria
   `c`, aprende-se um escalar `T_c > 0` e a confiança final é
   `sigmoid(logit_c / T_c)`. Uma temperatura por categoria porque VC-1,
   VC-2 e VC-3 têm dificuldades e frequências diferentes.
2. **Dados:** `T_c` é ajustado num split de **validação separado**, nunca
   nos dados de treino (que o modelo já decorou). O split é determinístico
   (seed fixa) para reprodutibilidade.
3. **Otimização:** minimiza-se a binary cross-entropy de validação em
   função de `T_c` por busca de seção áurea em `log T ∈ [ln 0.05, ln 20]`.
   A NLL é convexa em `1/T`, então a busca unidimensional converge para o
   ótimo global sem gradiente.
4. **Métrica:** Expected Calibration Error (ECE) com 10 bins de largura
   igual sobre todas as probabilidades (todas as categorias de todos os
   exemplos de validação). O pipeline de treino reporta ECE antes e
   depois da calibração.
5. **Limiar de aceite (AC-4):** ECE de validação ≤ 0.05. Só é verificável
   quando existir um dataset real; nos testes da 0.1.0 verificamos a
   corretude do método (recuperar uma temperatura conhecida), não o
   limiar.
6. As temperaturas são salvas junto dos pesos do modelo (mesmo arquivo),
   para que inferência e calibração nunca fiquem dessincronizadas.

## Alternativas consideradas

1. **Platt scaling (`a·logit + b`)** — mais flexível, mas o termo `b` pode
   deslocar a fronteira de decisão; temperature scaling preserva a ordem
   e a fronteira em 0.5 do logit, alterando só a "confiança".
2. **Isotonic regression** — não-paramétrico, exige muito mais dados de
   validação para não sobreajustar. Candidato quando houver dataset grande.
3. **Calibração dentro do treino (estilo RLCD do Jev)** — interessante, mas
   não documentado publicamente e fora de escopo; calibração pós-treino é
   o método estabelecido e auditável.

## Consequências

- Positivo: método simples, auditável e com um parâmetro por categoria.
- Negativo: temperature scaling não corrige erros de *ranking* — se o
  modelo ordena mal os exemplos, calibrar não melhora a acurácia.
- Risco: split de validação pequeno gera temperatura ruidosa. O pipeline
  reporta o tamanho do split para que isso seja visível.
