<?xml version="1.0" encoding="UTF-8"?>
<mxGraphModel dx="1200" dy="760" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1000" pageHeight="1500" math="0" shadow="0">
  <root>
    <mxCell id="0"/>
    <mxCell id="1" parent="0"/>
    <mxCell id="p1" value="INICIO" style="ellipse;whiteSpace=wrap;fillColor=#d5e8d4;strokeColor=#82b366;fontStyle=1;" vertex="1" parent="1"><mxGeometry x="340" y="20" width="120" height="40" as="geometry"/></mxCell>
    <mxCell id="p2" value="Inicializar:&#xa;π[s] = primera acción disponible&#xa;U[s] = 0   para todo s ∈ S" style="rounded=1;whiteSpace=wrap;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1"><mxGeometry x="250" y="100" width="300" height="65" as="geometry"/></mxCell>
    <mxCell id="pl1" value="── PASO 1: Evaluación de Política ──" style="text;html=1;align=center;verticalAlign=middle;resizable=0;autosize=1;strokeColor=none;fillColor=none;fontStyle=1;" vertex="1" parent="1"><mxGeometry x="250" y="205" width="260" height="28" as="geometry"/></mxCell>
    <mxCell id="p3" value="eval_method&#xa;= 'exact'?" style="rhombus;whiteSpace=wrap;fillColor=#fff2cc;strokeColor=#d6b656;" vertex="1" parent="1"><mxGeometry x="300" y="245" width="200" height="70" as="geometry"/></mxCell>
    <mxCell id="p4a" value="Construir sistema lineal (Ec. 16.14):&#xa;(I − γ·T_π)·U = R_π&#xa;Resolver con Eliminación Gaussiana" style="rounded=1;whiteSpace=wrap;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1"><mxGeometry x="80" y="355" width="300" height="65" as="geometry"/></mxCell>
    <mxCell id="p4b" value="Modified Policy Iteration:&#xa;Repetir k veces:&#xa;U(s) = Σ P(s'|s,π(s))·[R + γ·U(s')]" style="rounded=1;whiteSpace=wrap;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1"><mxGeometry x="420" y="355" width="300" height="65" as="geometry"/></mxCell>
    <mxCell id="p5" value="U ← solución obtenida (utilidades actualizadas)" style="rounded=1;whiteSpace=wrap;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1"><mxGeometry x="250" y="470" width="300" height="55" as="geometry"/></mxCell>
    <mxCell id="pl2" value="── PASO 2: Mejora de Política ──" style="text;html=1;align=center;verticalAlign=middle;resizable=0;autosize=1;strokeColor=none;fillColor=none;fontStyle=1;" vertex="1" parent="1"><mxGeometry x="250" y="560" width="260" height="28" as="geometry"/></mxCell>
    <mxCell id="p6" value="sin_cambios ← Verdadero" style="rounded=1;whiteSpace=wrap;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1"><mxGeometry x="250" y="595" width="300" height="55" as="geometry"/></mxCell>
    <mxCell id="p7" value="¿Quedan&#xa;estados sin&#xa;revisar?" style="rhombus;whiteSpace=wrap;fillColor=#fff2cc;strokeColor=#d6b656;" vertex="1" parent="1"><mxGeometry x="300" y="690" width="200" height="70" as="geometry"/></mxCell>
    <mxCell id="p8" value="Seleccionar siguiente estado s" style="rounded=1;whiteSpace=wrap;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1"><mxGeometry x="250" y="810" width="300" height="55" as="geometry"/></mxCell>
    <mxCell id="p9" value="a* ← argmax_a Q(s, a, U)" style="rounded=1;whiteSpace=wrap;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1"><mxGeometry x="250" y="905" width="300" height="55" as="geometry"/></mxCell>
    <mxCell id="p10" value="Q(s,a*,U) &gt;&#xa;Q(s,π[s],U) + ε?" style="rhombus;whiteSpace=wrap;fillColor=#fff2cc;strokeColor=#d6b656;" vertex="1" parent="1"><mxGeometry x="300" y="1005" width="200" height="70" as="geometry"/></mxCell>
    <mxCell id="p11" value="π[s] ← a*&#xa;sin_cambios ← Falso" style="rounded=1;whiteSpace=wrap;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1"><mxGeometry x="250" y="1125" width="300" height="55" as="geometry"/></mxCell>
    <mxCell id="p12" value="Guardar (π, U) en historial" style="rounded=1;whiteSpace=wrap;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1"><mxGeometry x="250" y="1230" width="300" height="55" as="geometry"/></mxCell>
    <mxCell id="p13" value="¿sin_cambios&#xa;= Verdadero?" style="rhombus;whiteSpace=wrap;fillColor=#fff2cc;strokeColor=#d6b656;" vertex="1" parent="1"><mxGeometry x="300" y="1330" width="200" height="70" as="geometry"/></mxCell>
    <mxCell id="p14" value="FIN&#xa;Retornar U, π*, historial" style="ellipse;whiteSpace=wrap;fillColor=#d5e8d4;strokeColor=#82b366;fontStyle=1;" vertex="1" parent="1"><mxGeometry x="320" y="1455" width="160" height="50" as="geometry"/></mxCell>
    <mxCell id="pe1" value="" style="endArrow=block;endFill=1;" edge="1" source="p1" target="p2" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="pe2" value="" style="endArrow=block;endFill=1;" edge="1" source="p2" target="p3" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="pe3a" value="Sí" style="endArrow=block;endFill=1;" edge="1" source="p3" target="p4a" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="pe3b" value="No" style="endArrow=block;endFill=1;" edge="1" source="p3" target="p4b" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="pe4a" value="" style="endArrow=block;endFill=1;" edge="1" source="p4a" target="p5" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="pe4b" value="" style="endArrow=block;endFill=1;" edge="1" source="p4b" target="p5" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="pe5" value="" style="endArrow=block;endFill=1;" edge="1" source="p5" target="p6" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="pe6" value="" style="endArrow=block;endFill=1;" edge="1" source="p6" target="p7" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="pe7" value="Sí" style="endArrow=block;endFill=1;" edge="1" source="p7" target="p8" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="pe8" value="" style="endArrow=block;endFill=1;" edge="1" source="p8" target="p9" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="pe9" value="" style="endArrow=block;endFill=1;" edge="1" source="p9" target="p10" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="pe10" value="Sí" style="endArrow=block;endFill=1;" edge="1" source="p10" target="p11" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="pe11" value="" style="endArrow=block;endFill=1;" edge="1" source="p11" target="p7" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="pe10b" value="No" style="endArrow=block;endFill=1;" edge="1" source="p10" target="p7" parent="1"><mxGeometry relative="1" as="geometry"><Array as="points"><mxPoint x="560" y="1040"/><mxPoint x="560" y="725"/></Array></mxGeometry></mxCell>
    <mxCell id="pe12" value="No" style="endArrow=block;endFill=1;" edge="1" source="p7" target="p12" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="pe13" value="" style="endArrow=block;endFill=1;" edge="1" source="p12" target="p13" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="pe14" value="Sí" style="endArrow=block;endFill=1;" edge="1" source="p13" target="p14" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="pe13b" value="No" style="endArrow=block;endFill=1;" edge="1" source="p13" target="p3" parent="1"><mxGeometry relative="1" as="geometry"><Array as="points"><mxPoint x="220" y="1365"/><mxPoint x="220" y="280"/></Array></mxGeometry></mxCell>
  </root>
</mxGraphModel>
