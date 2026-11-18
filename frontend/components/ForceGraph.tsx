"use client";

import ForceGraph2D, { type ForceGraphMethods } from "react-force-graph-2d";
import { useEffect, useRef, useState } from "react";

import { TYPE_COLORS, type GLink, type GNode } from "@/lib/graph";

export default function ForceGraph({
  data,
  selected,
  onSelect,
}: {
  data: { nodes: GNode[]; links: GLink[] };
  selected: number | null;
  onSelect: (n: GNode | null) => void;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fg = useRef<ForceGraphMethods<any, any> | undefined>(undefined);
  const fitted = useRef(false);

  useEffect(() => {
    fg.current?.d3Force("charge")?.strength(-90).distanceMax(220);
    fg.current?.d3Force("link")?.distance(55);
    fitted.current = false;
  }, [data]);
  const [size, setSize] = useState({ w: 800, h: 600 });

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const neighbours = new Set<number>();
  if (selected !== null) {
    for (const l of data.links) {
      const s = typeof l.source === "object" ? l.source.id : l.source;
      const t = typeof l.target === "object" ? l.target.id : l.target;
      if (s === selected) neighbours.add(t);
      if (t === selected) neighbours.add(s);
    }
  }

  return (
    <div ref={wrap} className="h-full w-full">
      <ForceGraph2D
        ref={fg}
        cooldownTicks={150}
        onEngineStop={() => {
          if (!fitted.current) fg.current?.zoomToFit(400, 40);
          fitted.current = true;
        }}
        width={size.w}
        height={size.h}
        graphData={data}
        backgroundColor="#09090b"
        nodeRelSize={4}
        nodeVal={(n: GNode) => 1 + n.degree}
        linkColor={() => "rgba(255,255,255,0.15)"}
        linkDirectionalArrowLength={3}
        linkDirectionalArrowRelPos={1}
        linkLabel={(l: GLink) => l.relation}
        onNodeClick={(n: GNode) => onSelect(n)}
        onBackgroundClick={() => onSelect(null)}
        nodeCanvasObject={(n: GNode, ctx, scale) => {
          const r = 3 + Math.sqrt(n.degree) * 2;
          const dim = selected !== null && n.id !== selected && !neighbours.has(n.id);
          ctx.globalAlpha = dim ? 0.2 : 1;
          ctx.beginPath();
          ctx.arc(n.x!, n.y!, r, 0, 2 * Math.PI);
          ctx.fillStyle = TYPE_COLORS[n.type] ?? "#94a3b8";
          ctx.fill();
          if (n.id === selected) {
            ctx.lineWidth = 2 / scale;
            ctx.strokeStyle = "#fff";
            ctx.stroke();
          }
          if (scale > 0.8 || n.degree > 1 || n.id === selected) {
            ctx.font = `${11 / scale}px system-ui`;
            ctx.fillStyle = "rgba(255,255,255,0.85)";
            ctx.textAlign = "center";
            ctx.fillText(n.name, n.x!, n.y! + r + 10 / scale);
          }
          ctx.globalAlpha = 1;
        }}
        nodePointerAreaPaint={(n: GNode, color, ctx) => {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(n.x!, n.y!, 6 + Math.sqrt(n.degree) * 2, 0, 2 * Math.PI);
          ctx.fill();
        }}
      />
    </div>
  );
}
