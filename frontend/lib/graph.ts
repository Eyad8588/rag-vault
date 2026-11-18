export type GNode = {
  id: number;
  name: string;
  type: string;
  degree: number;
  docs: { id: string; title: string; access_tag: string }[];
  x?: number;
  y?: number;
};
export type GLink = { source: number | GNode; target: number | GNode; relation: string; doc_id: string };

export const TYPE_COLORS: Record<string, string> = {
  person: "#f472b6",
  organization: "#60a5fa",
  location: "#34d399",
  product: "#fbbf24",
  technology: "#a78bfa",
  event: "#fb923c",
  concept: "#94a3b8",
};
