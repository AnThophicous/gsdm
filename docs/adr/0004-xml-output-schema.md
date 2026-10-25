# ADR-0004: Schema do XML de saída (v1)

- Status: Aceita
- Data: 2026-10-25

## Contexto

RF-4/RF-5 exigem que toda saída do GVF seja XML estruturado, sem texto
livre. Consumidores (CI, IDE, outros agentes via MCP) precisam de um
contrato estável e versionado.

## Decisão

Documento raiz único, versionado pelo atributo `schema-version`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<gvf-report schema-version="1" model="GVF" architecture="GSDM" architecture-version="0.1.0" threshold="0.5000">
  <file path="app/db.py">
    <assessment category="VC-1" name="sql-or-command-injection" confidence="0.9132" flagged="true">
      <location line="12" column="4"/>
    </assessment>
    <assessment category="VC-2" name="insecure-deserialization" confidence="0.0410" flagged="false">
      <location line="3" column="0"/>
    </assessment>
    <assessment category="VC-3" name="hardcoded-secret" confidence="0.0123" flagged="false">
      <location line="1" column="0"/>
    </assessment>
  </file>
  <skipped path="broken.py" reason="syntax-error"/>
</gvf-report>
```

Regras:

1. Todo `<file>` contém exatamente um `<assessment>` por categoria do
   escopo, na ordem VC-1, VC-2, VC-3 — o modelo sempre decide sobre todas
   (decisão tipada: booleano `flagged` + probabilidade `confidence`).
2. `confidence` é a probabilidade **calibrada** (ADR-0005), com 4 casas
   decimais, em [0, 1].
3. `flagged="true"` se e somente se `confidence >= threshold`.
4. `<location>` aponta o nó do grafo que mais pesou na decisão daquela
   categoria (ver ADR-0008, readout por atenção). É opcional no schema
   para permitir detectores futuros sem localização.
5. `<skipped>` registra arquivos que não puderam ser analisados, com
   `reason` em kebab-case de um vocabulário fechado (`syntax-error`).
6. Nenhum elemento carrega texto livre (conteúdo de nó). Toda informação
   está em atributos.
7. Mudança incompatível (remover/renomear atributo ou elemento) incrementa
   `schema-version`. Adição de atributo opcional não incrementa.

## Alternativas consideradas

1. **JSON** — mais comum em ferramentas modernas, mas o requisito do
   projeto é XML (decisão do usuário, SPEC G2).
2. **SARIF** (padrão de ferramentas de análise estática) — rico, mas é
   JSON e muito mais amplo que o necessário no MVP. Pode virar um
   presenter adicional no futuro sem tocar no domínio (ADR-0003).

## Consequências

- Positivo: contrato estável, trivial de consumir com qualquer parser XML.
- Positivo: o presenter é um adapter isolado — trocar ou adicionar formato
  (SARIF, JSON) não afeta use cases nem domínio.
- Pendente: publicar um XSD formal do v1 quando houver um consumidor
  externo que precise validar automaticamente.
