/**
 * Short, non-technical talk: how we grouped similar thumbnails.
 * Run: NODE_PATH=... node docs/cls_clustering_approach.js
 */
const pptxgen = require("pptxgenjs");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const UMAP = path.join(ROOT, "data/dinov3_cls_clusters/nopca-n15-mcs20-ms20/umap.png");
const UMAP_PEEL = path.join(ROOT, "data/dinov3_cls_clusters/nopca-n15-mcs20-ms20-r1/umap.png");

const pres = new pptxgen();
pres.defineLayout({ name: "WIDE16x9", width: 13.333, height: 7.5 });
pres.layout = "WIDE16x9";
pres.title = "Grouping similar thumbnails";
pres.author = "Olga Lyashevska";
pres.subject = "How we found visual types in 8,666 pictures";

const C = {
  navy: "1B2A4A",
  dk: "0F1B33",
  teal: "0E7C86",
  gold: "D4A843",
  off: "FAFAF7",
  wg: "E8E4DD",
  tx: "2C2C2C",
  mu: "6B7280",
  wh: "FFFFFF",
  lt: "E6F4F5",
  line: "D9D4C8",
};

const H = "Noto Serif";
const B = "Liberation Sans";
const W = 13.333;
const N = 10;

function goldBar(s) {
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: 0.08, fill: { color: C.gold },
  });
}

function footer(s, n) {
  s.addShape(pres.shapes.LINE, {
    x: 0.55, y: 7.08, w: W - 1.1, h: 0,
    line: { color: C.line, width: 0.75 },
  });
  s.addText("Grouping similar thumbnails", {
    x: 0.55, y: 7.16, w: 10.2, h: 0.24,
    fontFace: B, fontSize: 11, color: C.mu, margin: 0,
  });
  s.addText(`${n}  /  ${N}`, {
    x: 11.3, y: 7.16, w: 1.5, h: 0.24,
    fontFace: B, fontSize: 11, color: C.mu, align: "right", margin: 0,
  });
}

function eyebrow(s, text) {
  s.addText(text, {
    x: 0.55, y: 0.22, w: 12.2, h: 0.28,
    fontFace: B, fontSize: 12, color: C.gold, charSpacing: 2.2, margin: 0,
  });
}

function title(s, text, y = 0.52, h = 0.7) {
  s.addText(text, {
    x: 0.55, y, w: 12.2, h,
    fontFace: H, fontSize: 26, bold: true, color: C.navy, margin: 0, valign: "top",
  });
}

function card(s, x, y, w, h, fill) {
  s.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: fill || C.wh },
    shadow: { type: "outer", color: "000000", blur: 7, offset: 2, angle: 135, opacity: 0.08 },
  });
}

function accentBar(s, x, y, h, color) {
  s.addShape(pres.shapes.RECTANGLE, {
    x, y, w: 0.08, h, fill: { color: color || C.teal },
  });
}

// ── 1. Title ──────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.dk };
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: 0.08, fill: { color: C.gold },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 6.2, w: W, h: 1.3, fill: { color: C.navy },
  });

  s.addText("THUMBNAIL STUDY  ·  METHODS", {
    x: 0.75, y: 1.45, w: 11.8, h: 0.32,
    fontFace: B, fontSize: 13, color: C.gold, charSpacing: 2.5, margin: 0,
  });
  s.addText("Grouping similar\nthumbnails", {
    x: 0.75, y: 1.95, w: 11.8, h: 2.15,
    fontFace: H, fontSize: 40, bold: true, color: C.wh, margin: 0,
  });
  s.addText("Finding visual types in 8,666 pictures — without reading titles or captions", {
    x: 0.75, y: 4.35, w: 11.8, h: 0.45,
    fontFace: B, fontSize: 18, italic: true, color: C.gold, margin: 0,
  });
  s.addText("Olga Lyashevska    |    2020–2024 corpus    |    September 2026", {
    x: 0.75, y: 6.58, w: 11.8, h: 0.35,
    fontFace: B, fontSize: 14, color: C.wg, margin: 0,
  });
}

// ── 2. The pictures ───────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.off };
  goldBar(s);
  eyebrow(s, "THE PICTURES");
  title(s, "A balanced sample of video thumbnails");

  s.addText("We took the same number of videos from each of five years (2020–2024), downloaded their thumbnails, and kept only usable images: a real photo, the usual 640×360 size, not a tiny placeholder.", {
    x: 0.55, y: 1.3, w: 12.2, h: 0.75,
    fontFace: B, fontSize: 16, color: C.tx, margin: 0,
  });

  const stats = [
    { n: "12,500", l: "videos sampled", d: "2,500 from each year" },
    { n: "10,957", l: "thumbnails downloaded", d: "most pages had an image" },
    { n: "8,666", l: "pictures we analyse", d: "readable, correct size" },
  ];
  stats.forEach((st, i) => {
    const x = 0.55 + i * 4.2;
    card(s, x, 2.25, 4.0, 2.35);
    accentBar(s, x, 2.25, 2.35, i === 2 ? C.teal : C.gold);
    s.addText(st.n, {
      x: x + 0.28, y: 2.45, w: 3.5, h: 0.75,
      fontFace: H, fontSize: 32, bold: true, color: C.navy, margin: 0,
    });
    s.addText(st.l, {
      x: x + 0.28, y: 3.25, w: 3.5, h: 0.45,
      fontFace: B, fontSize: 16, bold: true, color: C.tx, margin: 0,
    });
    s.addText(st.d, {
      x: x + 0.28, y: 3.75, w: 3.5, h: 0.5,
      fontFace: B, fontSize: 14, color: C.mu, margin: 0,
    });
  });

  card(s, 0.55, 4.85, 12.2, 1.9, C.lt);
  s.addText("Before grouping, each picture is prepared the same way", {
    x: 0.8, y: 5.05, w: 11.7, h: 0.35,
    fontFace: B, fontSize: 15, bold: true, color: C.navy, margin: 0,
  });
  s.addText("Converted to colour, padded with black so a 16:9 frame becomes a square (the layout of the original shot is kept), then resized to 224×224 pixels so every picture is the same size.", {
    x: 0.8, y: 5.48, w: 11.7, h: 0.95,
    fontFace: B, fontSize: 15, color: C.tx, margin: 0,
  });

  footer(s, 2);
}

// ── 3. Fingerprints ───────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.off };
  goldBar(s);
  eyebrow(s, "STEP 1  ·  A FINGERPRINT FOR EACH PICTURE");
  title(s, "Turn each thumbnail into numbers");

  s.addText("We use DINOv3, a vision model trained on a huge collection of photos — not on our videos, and not on any labels we wrote. We never teach it our categories. We only ask: “what does this picture look like?”", {
    x: 0.55, y: 1.3, w: 12.2, h: 0.9,
    fontFace: B, fontSize: 16, color: C.tx, margin: 0,
  });

  const ideas = [
    {
      h: "A fingerprint, not a caption",
      t: "The model returns a long list of numbers for each image. Pictures that look alike get similar numbers. We never ask it “what is this?” in words.",
    },
    {
      h: "One summary per thumbnail",
      t: "We keep a single summary of the whole picture (composition, lighting, pose, branding). That is what we group. Local details in a crop are a separate track.",
    },
    {
      h: "The same picture always matches itself",
      t: "Given the same model and the same file, the numbers do not change. Any randomness later comes from how we arrange and group those numbers.",
    },
  ];
  ideas.forEach((idea, i) => {
    const x = 0.55 + i * 4.2;
    card(s, x, 2.4, 4.0, 4.25);
    s.addShape(pres.shapes.OVAL, {
      x: x + 0.25, y: 2.62, w: 0.5, h: 0.5, fill: { color: C.navy },
    });
    s.addText(String(i + 1), {
      x: x + 0.25, y: 2.62, w: 0.5, h: 0.5,
      fontFace: B, fontSize: 16, bold: true, color: C.gold,
      align: "center", valign: "middle", margin: 0,
    });
    s.addText(idea.h, {
      x: x + 0.25, y: 3.28, w: 3.5, h: 0.85,
      fontFace: H, fontSize: 18, color: C.navy, margin: 0,
    });
    s.addText(idea.t, {
      x: x + 0.25, y: 4.2, w: 3.5, h: 2.15,
      fontFace: B, fontSize: 15, color: C.tx, margin: 0,
    });
  });

  footer(s, 3);
}

// ── 4. How grouping works ─────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.off };
  goldBar(s);
  eyebrow(s, "STEP 2  ·  FROM FINGERPRINTS TO GROUPS");
  title(s, "Lay similar pictures near each other, then find clumps");

  s.addText("A list of a thousand numbers is hard to inspect. We do three simple things in order.", {
    x: 0.55, y: 1.28, w: 12.2, h: 0.45,
    fontFace: B, fontSize: 16, color: C.tx, margin: 0,
  });

  const steps = [
    {
      n: "1",
      h: "Arrange",
      t: "Put pictures on a map so that similar fingerprints sit close together. (The method is called UMAP.) Think of spreading photos on a table: twins land in the same corner.",
    },
    {
      n: "2",
      h: "Find clumps",
      t: "Look for dense patches on that map — groups of pictures that are close to each other and separated from the rest. (The method is called HDBSCAN.)",
    },
    {
      n: "3",
      h: "Allow leftovers",
      t: "A picture that does not sit in a clump stays ungrouped. That is deliberate. We would rather leave it out than force it into a type it does not match.",
    },
  ];
  steps.forEach((st, i) => {
    const y = 1.85 + i * 1.65;
    s.addShape(pres.shapes.OVAL, {
      x: 0.55, y: y + 0.25, w: 0.62, h: 0.62, fill: { color: C.navy },
    });
    s.addText(st.n, {
      x: 0.55, y: y + 0.25, w: 0.62, h: 0.62,
      fontFace: B, fontSize: 18, bold: true, color: C.gold,
      align: "center", valign: "middle", margin: 0,
    });
    card(s, 1.4, y, 11.35, 1.5);
    s.addText(st.h, {
      x: 1.65, y: y + 0.15, w: 10.9, h: 0.4,
      fontFace: B, fontSize: 18, bold: true, color: C.navy, margin: 0,
    });
    s.addText(st.t, {
      x: 1.65, y: y + 0.58, w: 10.9, h: 0.75,
      fontFace: B, fontSize: 15, color: C.tx, margin: 0,
    });
  });

  footer(s, 4);
}

// ── 5. Scores ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.off };
  goldBar(s);
  eyebrow(s, "SCORES");
  title(s, "Four abbreviations we will use");

  s.addText("A computer can also give a grouping a number. Two scores ask “is this grouping tight?” Two ask “do two groupings agree?” Higher is better for all four.", {
    x: 0.55, y: 1.28, w: 12.2, h: 0.65,
    fontFace: B, fontSize: 16, color: C.tx, margin: 0,
  });

  const mets = [
    {
      ab: "DBCV",
      full: "Density-Based Clustering Validation",
      t: "Are the clumps dense inside and separated from each other? Built for this kind of leftover-friendly grouping.",
    },
    {
      ab: "Silhouette",
      full: "Silhouette score",
      t: "For pictures that joined a group: is each one closer to its own group than to the next? Leftovers are ignored here.",
    },
    {
      ab: "ARI",
      full: "Adjusted Rand Index",
      t: "If we group twice, how often do the same pictures end up together? 1 = identical, 0 = no better than chance.",
    },
    {
      ab: "NMI",
      full: "Normalised Mutual Information",
      t: "How much does knowing one grouping tell you about the other? Also 0 to 1. A bit kinder than ARI when the number of groups differs.",
    },
  ];
  mets.forEach((m, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.55 + col * 6.35;
    const y = 2.05 + row * 2.4;
    card(s, x, y, 6.15, 2.22);
    s.addText(m.ab, {
      x: x + 0.28, y: y + 0.18, w: 5.6, h: 0.4,
      fontFace: H, fontSize: 22, bold: true, color: C.navy, margin: 0,
    });
    s.addText(m.full, {
      x: x + 0.28, y: y + 0.58, w: 5.6, h: 0.32,
      fontFace: B, fontSize: 13, italic: true, color: C.teal, margin: 0,
    });
    s.addText(m.t, {
      x: x + 0.28, y: y + 0.98, w: 5.6, h: 1.05,
      fontFace: B, fontSize: 15, color: C.tx, margin: 0,
    });
  });

  footer(s, 5);
}

// ── 6. Choosing settings ──────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.off };
  goldBar(s);
  eyebrow(s, "STEP 3  ·  CHOOSING A GROUPING");
  title(s, "We tried many settings, then looked at the pictures");

  const qs = [
    {
      h: "How local is “similar”?",
      t: "Should a picture match only its nearest neighbours, or a wider circle?",
    },
    {
      h: "How small may a group be?",
      t: "A group of 3 is probably a fluke. A group of 2,000 is probably not a type.",
    },
    {
      h: "How picky is membership?",
      t: "Stricter rules make tighter groups and more leftovers. Looser rules swallow mixed pictures.",
    },
  ];
  qs.forEach((q, i) => {
    const x = 0.55 + i * 4.2;
    card(s, x, 1.35, 4.0, 2.35);
    s.addText(q.h, {
      x: x + 0.25, y: 1.55, w: 3.5, h: 0.7,
      fontFace: H, fontSize: 18, color: C.navy, margin: 0,
    });
    s.addText(q.t, {
      x: x + 0.25, y: 2.3, w: 3.5, h: 1.15,
      fontFace: B, fontSize: 15, color: C.tx, margin: 0,
    });
  });

  s.addText("We crossed those settings into 99 combinations. Each cell got a DBCV score, a silhouette score, and a mark for how many pictures were left out. The automatic winner was always the same kind of answer: split everything into two giant groups with nothing left over. That is a split of the map, not a list of thumbnail types — so we discarded it.", {
    x: 0.55, y: 3.9, w: 12.2, h: 1.05,
    fontFace: B, fontSize: 15, color: C.tx, margin: 0,
  });

  card(s, 0.55, 5.1, 12.2, 1.65, C.navy);
  s.addText("What we kept instead", {
    x: 0.8, y: 5.25, w: 11.7, h: 0.32,
    fontFace: B, fontSize: 14, bold: true, color: C.gold, margin: 0,
  });
  s.addText("A grouping with a reviewable number of groups (tens, not two and not hundreds), leftover pictures to inspect later, and sample grids that look consistent. Looking at the pictures is the test that matters.", {
    x: 0.8, y: 5.62, w: 11.7, h: 0.9,
    fontFace: B, fontSize: 15, color: C.wh, margin: 0,
  });

  footer(s, 6);
}

// ── 7. Results + UMAP ─────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.off };
  goldBar(s);
  eyebrow(s, "WHAT WE GOT");
  title(s, "33 groups, and many pictures left ungrouped", 0.48, 0.55);

  const kpis = [
    { n: "33", l: "groups" },
    { n: "41%", l: "ungrouped" },
    { n: "0.27", l: "DBCV" },
    { n: "0.54", l: "silhouette" },
  ];
  kpis.forEach((k, i) => {
    const x = 0.55 + i * 1.55;
    s.addText(k.n, {
      x, y: 1.12, w: 1.5, h: 0.48,
      fontFace: H, fontSize: 24, bold: true, color: C.navy, margin: 0,
    });
    s.addText(k.l, {
      x, y: 1.58, w: 1.5, h: 0.5,
      fontFace: B, fontSize: 13, color: C.mu, margin: 0,
    });
  });

  s.addText("Each colour is a group. Grey points did not join one. The isolated island on the left is a real clump, not a drawing glitch. This map is only a picture of the grouping — we decide what a group means by looking at example thumbnails, not at the dots.", {
    x: 0.55, y: 2.15, w: 6.2, h: 1.7,
    fontFace: B, fontSize: 15, color: C.tx, margin: 0,
  });

  const uw = 5.85;
  const uh = uw * (1280 / 1600);
  s.addImage({
    path: UMAP,
    x: 7.0, y: 1.1, w: uw, h: uh,
    sizing: { type: "contain", w: uw, h: uh },
  });

  card(s, 0.55, 4.05, 6.2, 2.55, C.lt);
  s.addText("Not every “group” is a type", {
    x: 0.75, y: 4.22, w: 5.8, h: 0.35,
    fontFace: B, fontSize: 15, bold: true, color: C.navy, margin: 0,
  });
  s.addText("Two large groups (about 740 and 840 pictures) are mixed leftover piles. We do not treat those as types — same as the ungrouped pictures. The other groups are the candidates for named visual types.", {
    x: 0.75, y: 4.65, w: 5.8, h: 1.7,
    fontFace: B, fontSize: 15, color: C.tx, margin: 0,
  });

  footer(s, 7);
}

// ── 8. Stability ──────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.off };
  goldBar(s);
  eyebrow(s, "A CAVEAT");
  title(s, "The groups are a useful look, not a unique catalogue");

  s.addText("The arranging step has a random starting point. We froze the settings and ran the grouping 100 times, only changing that start. We then compared every pair of runs with ARI and NMI. If the types were unique, both would sit near 1.", {
    x: 0.55, y: 1.3, w: 12.2, h: 0.85,
    fontFace: B, fontSize: 16, color: C.tx, margin: 0,
  });

  const cols = [
    {
      h: "What we saw",
      t: "Mean ARI was 0.47, mean NMI 0.57 — and both jumped around a lot. Some repeats looked like our main grouping; others collapsed into a few big clumps.",
    },
    {
      h: "What that means",
      t: "There is real structure in the pictures, but it is not one official list of 33 kinds. A different random start can draw the boundaries differently.",
    },
    {
      h: "How we use it",
      t: "We inspect one run (the map on the previous slide) and base what we say on those example pictures. We do not say “there are exactly 33 kinds of thumbnail.”",
    },
  ];
  cols.forEach((c, i) => {
    const x = 0.55 + i * 4.2;
    card(s, x, 2.35, 4.0, 4.3);
    accentBar(s, x, 2.35, 4.3, i === 2 ? C.gold : C.teal);
    s.addText(c.h, {
      x: x + 0.28, y: 2.55, w: 3.5, h: 0.55,
      fontFace: H, fontSize: 20, color: C.navy, margin: 0,
    });
    s.addText(c.t, {
      x: x + 0.28, y: 3.2, w: 3.5, h: 3.1,
      fontFace: B, fontSize: 16, color: C.tx, margin: 0,
    });
  });

  footer(s, 8);
}

// ── 9. Leftovers ──────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.off };
  goldBar(s);
  eyebrow(s, "LEFTOVERS");
  title(s, "We look again at pictures that did not join a group", 0.48, 0.55);

  s.addText("Ungrouped does not mean “nothing here.” We take those leftovers and run the same grouping on that subset only — new map, same rules. We did this twice.", {
    x: 0.55, y: 1.12, w: 6.25, h: 1.0,
    fontFace: B, fontSize: 15, color: C.tx, margin: 0,
  });

  const rounds = [
    { r: "First pass", n: "8,666", d: "33 groups  ·  3,558 left out" },
    { r: "Look again", n: "3,558", d: "11 extra groups  ·  1,254 still out" },
    { r: "Once more", n: "1,254", d: "9 extra groups  ·  632 still out" },
  ];
  rounds.forEach((r, i) => {
    const y = 2.25 + i * 1.35;
    s.addShape(pres.shapes.OVAL, {
      x: 0.55, y: y + 0.12, w: 0.7, h: 0.7, fill: { color: C.navy },
    });
    s.addText(String(i + 1), {
      x: 0.55, y: y + 0.12, w: 0.7, h: 0.7,
      fontFace: B, fontSize: 16, bold: true, color: C.gold,
      align: "center", valign: "middle", margin: 0,
    });
    if (i < 2) {
      s.addShape(pres.shapes.LINE, {
        x: 0.9, y: y + 0.85, w: 0, h: 0.58,
        line: { color: C.gold, width: 1.5 },
      });
    }
    s.addText(r.r + "  ·  " + r.n + " pictures", {
      x: 1.45, y: y, w: 5.3, h: 0.38,
      fontFace: H, fontSize: 16, color: C.navy, margin: 0,
    });
    s.addText(r.d, {
      x: 1.45, y: y + 0.42, w: 5.3, h: 0.5,
      fontFace: B, fontSize: 15, color: C.tx, margin: 0,
    });
  });

  const pw = 5.7;
  const ph = pw * (1280 / 1600);
  s.addImage({
    path: UMAP_PEEL,
    x: 7.05, y: 1.12, w: pw, h: ph,
    sizing: { type: "contain", w: pw, h: ph },
  });
  s.addText("Second map, leftovers only. The large salmon clump is a mixed pile, not a type.", {
    x: 7.05, y: 1.12 + ph + 0.08, w: 5.7, h: 0.5,
    fontFace: B, fontSize: 13, italic: true, color: C.mu, margin: 0,
  });

  footer(s, 9);
}

// ── 10. Takeaways ─────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.dk };
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: 0.08, fill: { color: C.gold },
  });
  s.addText("IN SHORT", {
    x: 0.75, y: 0.45, w: 11.8, h: 0.3,
    fontFace: B, fontSize: 13, color: C.gold, charSpacing: 2.5, margin: 0,
  });
  s.addText("What to take from this", {
    x: 0.75, y: 0.85, w: 11.8, h: 0.55,
    fontFace: H, fontSize: 28, bold: true, color: C.wh, margin: 0,
  });

  const pts = [
    {
      n: "01",
      t: "Pictures can be grouped by look alone",
      d: "A vision model gives each thumbnail a fingerprint. We never use titles or captions for this step.",
    },
    {
      n: "02",
      t: "We kept a reviewable set of groups",
      d: "33 groups on the first pass (DBCV 0.27, silhouette 0.54), judged by looking at example pictures. Automatic scores that split everything in two were ignored.",
    },
    {
      n: "03",
      t: "Leftovers are part of the method",
      d: "About 4 in 10 pictures sit out at first. Grouping them again finds some extra structure and some mixed piles. About 7% stay ungrouped.",
    },
    {
      n: "04",
      t: "Treat types as examples, not a census",
      d: "Repeating the grouping with a different random start changes the boundaries (ARI 0.47, NMI 0.57). Claims about visual types should rest on the pictures we inspected.",
    },
  ];
  pts.forEach((p, i) => {
    const y = 1.6 + i * 1.28;
    s.addText(p.n, {
      x: 0.75, y, w: 0.7, h: 0.4,
      fontFace: B, fontSize: 16, bold: true, color: C.gold, margin: 0,
    });
    s.addText(p.t, {
      x: 1.55, y, w: 10.8, h: 0.38,
      fontFace: B, fontSize: 18, bold: true, color: C.wh, margin: 0,
    });
    s.addText(p.d, {
      x: 1.55, y: y + 0.42, w: 10.8, h: 0.65,
      fontFace: B, fontSize: 15, color: C.wg, margin: 0,
    });
  });
}

pres.writeFile({ fileName: path.join(__dirname, "cls_clustering_approach.pptx") })
  .then(() => console.log("Wrote docs/cls_clustering_approach.pptx"))
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
