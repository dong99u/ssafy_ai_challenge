"use strict";

const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");
PptxGenJS.ShapeType = PptxGenJS.ShapeType || new PptxGenJS().ShapeType;
const {
  autoFontSize,
  calcTextBoxHeightSimple,
  imageSizingContain,
  svgToDataUri,
  safeOuterShadow,
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers");

const OUT_DIR = path.join(__dirname, "dist");
const OUT_FILE = path.join(OUT_DIR, "ssafy-ai-challenge-strategy.pptx");
const COVER_ASSET_PATH = path.join(__dirname, "..", "assets", "cover-hero-v1.png");

const FONT_HEAD = "Apple SD Gothic Neo";
const FONT_BODY = "Apple SD Gothic Neo";

const COLORS = {
  bg: "F5F5F0",
  ink: "12212E",
  muted: "62707C",
  line: "D8DED9",
  teal: "0E7490",
  green: "2E7D5A",
  lime: "90A955",
  amber: "E79A3B",
  coral: "C75C3A",
  navy: "163243",
  white: "FFFFFF",
  soft: "E9EEEA",
  paleTeal: "D9EEF2",
  paleGreen: "E3F0E9",
  paleAmber: "F9E8D2",
  paleCoral: "F6DDD7",
};

const PPT_W = 13.333;
const PPT_H = 7.5;

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function fitText(text, opts = {}) {
  return autoFontSize(text, opts.fontFace || FONT_BODY, {
    minFontSize: opts.minFontSize || 10,
    maxFontSize: opts.maxFontSize || 28,
    fontSize: opts.fontSize || opts.maxFontSize || 18,
    margin: opts.margin || 0,
    breakLine: false,
    valign: opts.valign || "top",
    ...opts,
  });
}

function addText(slide, text, opts = {}) {
  const fontFace = opts.fontFace || FONT_BODY;
  const fontSize =
    opts.fontSize || fitText(text, { ...opts, fontFace }).fontSize;
  slide.addText(text, {
    fontFace,
    fontSize,
    color: COLORS.ink,
    margin: 0,
    breakLine: false,
    valign: "top",
    ...opts,
  });
}

function addRichText(slide, runs, opts = {}) {
  slide.addText(runs, {
    fontFace: FONT_BODY,
    color: COLORS.ink,
    margin: 0,
    breakLine: false,
    valign: "top",
    ...opts,
  });
}

function addSlideFrame(slide, section, title, subtitle) {
  slide.background = { color: COLORS.bg };
  slide.addShape(PptxGenJS.ShapeType.roundRect, {
    x: 0.55,
    y: 0.38,
    w: 1.2,
    h: 0.36,
    rectRadius: 0.08,
    fill: { color: COLORS.navy },
    line: { color: COLORS.navy },
  });
  addText(slide, section, {
    x: 0.75,
    y: 0.44,
    w: 0.8,
    h: 0.18,
    fontFace: FONT_HEAD,
    fontSize: 10.5,
    bold: true,
    color: COLORS.white,
    align: "center",
  });
  addText(slide, title, {
    x: 0.72,
    y: 0.9,
    w: 7.6,
    h: 0.42,
    fontFace: FONT_HEAD,
    maxFontSize: 24,
    minFontSize: 18,
    bold: true,
    color: COLORS.ink,
  });
  if (subtitle) {
    addText(slide, subtitle, {
      x: 0.75,
      y: 1.5,
      w: 8.2,
      h: 0.26,
      fontFace: FONT_BODY,
      maxFontSize: 11.5,
      minFontSize: 10,
      color: COLORS.muted,
    });
  }
  slide.addShape(PptxGenJS.ShapeType.line, {
    x: 0.72,
    y: 1.94,
    w: 11.85,
    h: 0,
    line: { color: COLORS.line, width: 1.4 },
  });
}

function addPanel(slide, x, y, w, h, fill = COLORS.white, line = COLORS.line) {
  slide.addShape(PptxGenJS.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.12,
    fill: { color: fill },
    line: { color: line, width: 1 },
    shadow: safeOuterShadow("000000", 0.12, 45, 1.5, 0.5),
  });
}

function addMetricCard(slide, x, y, w, h, label, value, detail, accent) {
  addPanel(slide, x, y, w, h, COLORS.white, COLORS.line);
  slide.addShape(PptxGenJS.ShapeType.roundRect, {
    x: x + 0.18,
    y: y + 0.18,
    w: 0.18,
    h: h - 0.36,
    rectRadius: 0.05,
    fill: { color: accent },
    line: { color: accent },
  });
  addText(slide, label, {
    x: x + 0.48,
    y: y + 0.18,
    w: w - 0.66,
    h: 0.22,
    fontSize: 9.5,
    bold: true,
    color: COLORS.muted,
  });
  addText(slide, value, {
    x: x + 0.48,
    y: y + 0.5,
    w: w - 0.55,
    h: 0.36,
    fontFace: FONT_HEAD,
    fontSize: 22,
    bold: true,
    color: COLORS.ink,
  });
  addText(slide, detail, {
    x: x + 0.48,
    y: y + 0.96,
    w: w - 0.6,
    h: 0.36,
    fontSize: 9.5,
    color: COLORS.muted,
  });
}

function addBulletBlock(slide, x, y, w, title, items, accent = COLORS.teal) {
  addText(slide, title, {
    x,
    y,
    w,
    h: 0.24,
    fontSize: 12,
    fontFace: FONT_HEAD,
    bold: true,
    color: COLORS.ink,
  });
  const bulletX = x + 0.02;
  let cursorY = y + 0.34;
  for (const item of items) {
    slide.addShape(PptxGenJS.ShapeType.ellipse, {
      x: bulletX,
      y: cursorY + 0.06,
      w: 0.08,
      h: 0.08,
      fill: { color: accent },
      line: { color: accent },
    });
    const boxH = Math.max(0.3, calcTextBoxHeightSimple(10.2, 2, 1.15, 0.08));
    addText(slide, item, {
      x: bulletX + 0.15,
      y: cursorY,
      w: w - 0.18,
      h: boxH,
      fontSize: 10.2,
      color: COLORS.ink,
    });
    cursorY += boxH + 0.12;
  }
}

function addSimpleBullets(slide, x, y, w, items, accent = COLORS.teal, fontSize = 10.2) {
  const bulletX = x + 0.02;
  let cursorY = y;
  for (const item of items) {
    slide.addShape(PptxGenJS.ShapeType.ellipse, {
      x: bulletX,
      y: cursorY + 0.08,
      w: 0.08,
      h: 0.08,
      fill: { color: accent },
      line: { color: accent },
    });
    const boxH = Math.max(0.28, calcTextBoxHeightSimple(fontSize, 2, 1.12, 0.04));
    addText(slide, item, {
      x: bulletX + 0.15,
      y: cursorY,
      w: w - 0.18,
      h: boxH,
      fontSize,
      color: COLORS.ink,
    });
    cursorY += boxH + 0.08;
  }
}

function addTag(slide, x, y, w, text, fill, color = COLORS.ink) {
  slide.addShape(PptxGenJS.ShapeType.roundRect, {
    x,
    y,
    w,
    h: 0.28,
    rectRadius: 0.08,
    fill: { color: fill },
    line: { color: fill },
  });
  addText(slide, text, {
    x,
    y: y + 0.05,
    w,
    h: 0.14,
    fontSize: 8.5,
    bold: true,
    color,
    align: "center",
  });
}

function addComparisonRow(slide, y, label, base, target, note = "", accent = COLORS.teal) {
  addText(slide, label, {
    x: 0.95,
    y,
    w: 1.2,
    h: 0.18,
    fontSize: 10,
    bold: true,
    color: COLORS.ink,
  });
  addText(slide, base, {
    x: 2.4,
    y,
    w: 1.2,
    h: 0.18,
    fontSize: 10,
    color: COLORS.muted,
    align: "center",
  });
  addText(slide, target, {
    x: 4.1,
    y,
    w: 1.5,
    h: 0.18,
    fontSize: 10,
    bold: true,
    color: accent,
    align: "center",
  });
  if (note) {
    addText(slide, note, {
      x: 5.95,
      y: y - 0.01,
      w: 5.95,
      h: 0.26,
      fontSize: 9.5,
      color: COLORS.ink,
    });
  }
}

function addBar(slide, x, y, w, h, fraction, fill, bg = COLORS.soft) {
  slide.addShape(PptxGenJS.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.08,
    fill: { color: bg, transparency: 100 },
    line: { color: bg, width: 1 },
  });
  slide.addShape(PptxGenJS.ShapeType.roundRect, {
    x,
    y,
    w: Math.max(0.12, w * fraction),
    h,
    rectRadius: 0.08,
    fill: { color: fill },
    line: { color: fill },
  });
}

function heroSvg() {
  return svgToDataUri(`
    <svg xmlns="http://www.w3.org/2000/svg" width="900" height="560" viewBox="0 0 900 560">
      <defs>
        <linearGradient id="g1" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#0E7490"/>
          <stop offset="100%" stop-color="#90A955"/>
        </linearGradient>
      </defs>
      <rect x="14" y="14" width="872" height="532" rx="36" fill="#FFFFFF" stroke="#D8DED9" stroke-width="8"/>
      <circle cx="450" cy="280" r="108" fill="#F5F5F0" stroke="url(#g1)" stroke-width="14"/>
      <path d="M360 244c32-64 112-88 176-52" fill="none" stroke="#0E7490" stroke-width="22" stroke-linecap="round"/>
      <path d="M529 168l36 18-18 36" fill="none" stroke="#0E7490" stroke-width="22" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M548 318c-34 64-114 86-176 48" fill="none" stroke="#90A955" stroke-width="22" stroke-linecap="round"/>
      <path d="M375 392l-36-18 18-36" fill="none" stroke="#90A955" stroke-width="22" stroke-linecap="round" stroke-linejoin="round"/>
      <rect x="394" y="214" width="112" height="86" rx="18" fill="#163243"/>
      <circle cx="450" cy="257" r="18" fill="#F5F5F0"/>
      <rect x="426" y="303" width="48" height="18" rx="9" fill="#163243"/>
      <rect x="596" y="150" width="170" height="74" rx="24" fill="#F9E8D2"/>
      <text x="681" y="195" font-family="Apple SD Gothic Neo" font-size="34" font-weight="700" text-anchor="middle" fill="#C75C3A">a / b / c / d ?</text>
      <rect x="120" y="340" width="188" height="110" rx="28" fill="#D9EEF2"/>
      <rect x="150" y="362" width="44" height="54" rx="10" fill="#0E7490"/>
      <rect x="210" y="356" width="58" height="68" rx="16" fill="#90A955"/>
      <rect x="278" y="376" width="20" height="40" rx="10" fill="#E79A3B"/>
      <text x="214" y="445" font-family="Apple SD Gothic Neo" font-size="28" font-weight="700" text-anchor="middle" fill="#163243">재활용 이미지</text>
    </svg>
  `);
}

function permutationChipSvg(labels, accent) {
  const xs = [26, 96, 166, 236];
  const texts = labels
    .map(
      (label, i) => `
      <rect x="${xs[i]}" y="18" width="50" height="50" rx="16" fill="#FFFFFF" stroke="${accent}" stroke-width="4"/>
      <text x="${xs[i] + 25}" y="50" font-family="Apple SD Gothic Neo" font-size="28" font-weight="700" text-anchor="middle" fill="${accent}">${label}</text>
    `
    )
    .join("");
  return svgToDataUri(`
    <svg xmlns="http://www.w3.org/2000/svg" width="312" height="86" viewBox="0 0 312 86">
      <rect x="2" y="2" width="308" height="82" rx="24" fill="#F5F5F0" stroke="#D8DED9" stroke-width="3"/>
      ${texts}
    </svg>
  `);
}

function addAudit(slide, pptx) {
  warnIfSlideHasOverlaps(slide, pptx, { ignoreDecorativeShapes: true });
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

function buildDeck() {
  const pptx = new PptxGenJS();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "OpenAI Codex";
  pptx.company = "SSAFY";
  pptx.subject = "SSAFY 15기 AI Challenge 전략 발표";
  pptx.title = "SSAFY 15기 AI Challenge 전략";
  pptx.lang = "ko-KR";
  pptx.theme = {
    headFontFace: FONT_HEAD,
    bodyFontFace: FONT_BODY,
    lang: "ko-KR",
  };
  pptx.defineSlideMaster({
    title: "MASTER_WIDE",
    background: { color: COLORS.bg },
    objects: [],
    slideNumber: {
      x: 12.45,
      y: 7.04,
      w: 0.45,
      h: 0.18,
      color: COLORS.muted,
      fontFace: FONT_BODY,
      fontSize: 8,
      align: "right",
    },
  });
  pptx.masterSlideName = "MASTER_WIDE";
  pptx.writeOptions = { compression: true };

  // Slide 1
  {
    const slide = pptx.addSlide("MASTER_WIDE");
    slide.background = { color: COLORS.bg };
    slide.addShape(PptxGenJS.ShapeType.roundRect, {
      x: 0.65,
      y: 0.62,
      w: 1.9,
      h: 0.36,
      rectRadius: 0.08,
      fill: { color: COLORS.navy },
      line: { color: COLORS.navy },
    });
    addText(slide, "TEAM PRESENTATION", {
      x: 0.8,
      y: 0.68,
      w: 1.6,
      h: 0.16,
      fontSize: 9.8,
      bold: true,
      color: COLORS.white,
      align: "center",
    });
    addText(slide, "SSAFY 15기 AI Challenge", {
      x: 0.7,
      y: 1.2,
      w: 5.2,
      h: 0.34,
      fontFace: FONT_HEAD,
      fontSize: 21,
      bold: true,
    });
    addText(slide, "재활용품 이미지 VQA 최종 전략 발표", {
      x: 0.72,
      y: 1.78,
      w: 5.45,
      h: 0.42,
      fontFace: FONT_HEAD,
      fontSize: 24,
      bold: true,
      color: COLORS.ink,
    });
    addText(slide, "Qwen2.5-VL 7B, QLoRA, full-data 학습, logprob 추론, 선택지 셔플 TTA로 구성한 최종 제출 전략을 공유합니다.", {
      x: 0.72,
      y: 2.42,
      w: 5.15,
      h: 0.56,
      fontSize: 11.2,
      color: COLORS.muted,
    });
    addPanel(slide, 0.72, 3.3, 3.2, 1.02, COLORS.paleTeal, COLORS.paleTeal);
    addText(slide, "출발! 드림팀", {
      x: 0.96,
      y: 3.56,
      w: 2.72,
      h: 0.26,
      fontFace: FONT_HEAD,
      fontSize: 22,
      bold: true,
      color: COLORS.navy,
      align: "center",
    });
    addText(slide, "대전 6반", {
      x: 0.76,
      y: 4.62,
      w: 0.84,
      h: 0.18,
      fontSize: 11,
      fontFace: FONT_HEAD,
      bold: true,
      color: COLORS.teal,
    });
    addText(slide, "이강륜 · 윤병현 · 박동규 · 김영호", {
      x: 0.76,
      y: 4.95,
      w: 4.8,
      h: 0.26,
      fontSize: 13,
      fontFace: FONT_HEAD,
      bold: true,
      color: COLORS.ink,
    });
    addTag(slide, 0.76, 5.66, 1.18, "Scale-up", COLORS.paleTeal, COLORS.teal);
    addTag(slide, 2.04, 5.66, 1.28, "Full-data", COLORS.paleGreen, COLORS.green);
    addTag(slide, 3.44, 5.66, 1.18, "Logprob", COLORS.paleAmber, COLORS.amber);
    addTag(slide, 4.74, 5.66, 0.92, "TTA", COLORS.paleCoral, COLORS.coral);
    slide.addShape(PptxGenJS.ShapeType.line, {
      x: 0.76,
      y: 6.42,
      w: 4.9,
      h: 0,
      line: { color: COLORS.line, width: 1.2 },
    });
    addText(slide, "SSAFY 15기 AI Challenge Final Strategy", {
      x: 0.76,
      y: 6.58,
      w: 4.9,
      h: 0.18,
      fontSize: 10,
      color: COLORS.muted,
    });
    slide.addImage({
      path: COVER_ASSET_PATH,
      ...imageSizingContain(COVER_ASSET_PATH, 6.45, 0.96, 6.22, 5.72),
    });
    addPanel(slide, 7.42, 5.94, 4.24, 0.58, COLORS.white, COLORS.line);
    addText(slide, "작은 시각 단서를 더 잘 읽고, 선택지 편향을 더 적게 믿는 방향으로 전략을 다듬었습니다.", {
      x: 7.68,
      y: 6.14,
      w: 3.72,
      h: 0.18,
      fontSize: 10.6,
      fontFace: FONT_HEAD,
      bold: true,
      color: COLORS.navy,
      align: "center",
    });
    addAudit(slide, pptx);
  }

  // Slide 2
  {
    const slide = pptx.addSlide("MASTER_WIDE");
    addSlideFrame(slide, "01", "문제 정의와 승부 조건", "정답률 게임이기 때문에 모델 자체보다도 데이터 활용 방식과 추론 안정성이 직접 점수로 이어졌습니다.");
    addPanel(slide, 0.82, 2.08, 4.05, 4.65);
    addText(slide, "대회 입력 / 출력", {
      x: 1.05,
      y: 2.34,
      w: 2.2,
      h: 0.22,
      fontSize: 13,
      fontFace: FONT_HEAD,
      bold: true,
    });
    addBulletBlock(slide, 1.05, 2.72, 3.5, "문제", [
      "입력: 재활용품 이미지 + 한국어 4지선다 질문",
      "출력: a / b / c / d 중 하나의 답",
      "평가지표: Accuracy",
    ], COLORS.teal);
    addText(slide, "데이터 규모", {
      x: 1.05,
      y: 4.68,
      w: 1.6,
      h: 0.2,
      fontSize: 12,
      fontFace: FONT_HEAD,
      bold: true,
    });
    const counts = [
      ["train", 5073, COLORS.teal],
      ["dev", 4413, COLORS.green],
      ["test", 5074, COLORS.amber],
    ];
    let y = 5.16;
    for (const [name, value, color] of counts) {
      addText(slide, name, { x: 1.05, y, w: 0.6, h: 0.16, fontSize: 10.5, bold: true });
      addBar(slide, 1.75, y + 0.02, 2.3, 0.18, value / 5200, color);
      addText(slide, value.toLocaleString("ko-KR"), {
        x: 4.15,
        y: y - 0.01,
        w: 0.5,
        h: 0.18,
        fontSize: 10.5,
        bold: true,
        align: "right",
      });
      y += 0.44;
    }
    addPanel(slide, 5.1, 2.08, 7.35, 4.65, COLORS.white, COLORS.line);
    addText(slide, "무엇이 어려웠나", {
      x: 5.45,
      y: 2.34,
      w: 2.0,
      h: 0.22,
      fontSize: 13,
      fontFace: FONT_HEAD,
      bold: true,
    });
    const painPoints = [
      ["세부 시각 단서", "라벨, 재질, 광택, 찌그러짐처럼 작은 단서가 정답을 가름"],
      ["한국어 객관식", "질문 해석 + 보기 비교 + 문자 하나만 출력해야 함"],
      ["선택지 편향", "VLM이 특정 위치의 보기를 선호하면 accuracy가 바로 깎임"],
    ];
    let py = 2.84;
    for (const [title, detail] of painPoints) {
      addPanel(slide, 5.45, py, 6.65, 0.96, COLORS.soft, COLORS.soft);
      addText(slide, title, {
        x: 5.72,
        y: py + 0.18,
        w: 1.22,
        h: 0.16,
        fontSize: 11,
        fontFace: FONT_HEAD,
        bold: true,
      });
      addText(slide, detail, {
        x: 7.26,
        y: py + 0.17,
        w: 4.55,
        h: 0.36,
        fontSize: 9.8,
        color: COLORS.ink,
      });
      py += 1.18;
    }
    addText(slide, "결론: 학습 성능 + 추론 안정화 + 데이터 확대를 동시에 잡는 전략이 필요했습니다.", {
      x: 5.45,
      y: 6.3,
      w: 6.65,
      h: 0.26,
      fontSize: 11.3,
      fontFace: FONT_HEAD,
      bold: true,
      color: COLORS.navy,
    });
    addAudit(slide, pptx);
  }

  // Slide 3
  {
    const slide = pptx.addSlide("MASTER_WIDE");
    addSlideFrame(slide, "02", "최종 전략 한 장 요약", "성능 향상은 한 번의 묘수가 아니라 네 개의 작은 결정이 누적된 결과였습니다.");
    const cards = [
      {
        x: 0.85,
        y: 2.2,
        w: 2.85,
        title: "Model scale-up",
        accent: COLORS.teal,
        fill: COLORS.paleTeal,
        lines: ["Qwen2.5-VL 3B → 7B", "384px → 672px", "H100 80GB로 학습 여유 확보"],
      },
      {
        x: 3.97,
        y: 2.2,
        w: 2.85,
        title: "Fine-tuning tuning",
        accent: COLORS.green,
        fill: COLORS.paleGreen,
        lines: ["LoRA r=8 → 64", "LR 1e-4 → 1e-5", "3 epoch + cosine scheduler"],
      },
      {
        x: 7.09,
        y: 2.2,
        w: 2.85,
        title: "Data expansion",
        accent: COLORS.amber,
        fill: COLORS.paleAmber,
        lines: ["train + dev 전체 학습", "dev는 5인 다수결로 gold 생성", "9,486개까지 학습 데이터 확대"],
      },
      {
        x: 10.21,
        y: 2.2,
        w: 2.25,
        title: "Inference control",
        accent: COLORS.coral,
        fill: COLORS.paleCoral,
        lines: ["logprob 직접 추출", "선택지 셔플 TTA", "앙상블은 자동선택 후 기각"],
      },
    ];
    for (const card of cards) {
      addPanel(slide, card.x, card.y, card.w, 2.98, card.fill, card.fill);
      slide.addShape(PptxGenJS.ShapeType.roundRect, {
        x: card.x + 0.22,
        y: card.y + 0.22,
        w: card.w - 0.44,
        h: 0.32,
        rectRadius: 0.08,
        fill: { color: card.accent },
        line: { color: card.accent },
      });
      addText(slide, card.title, {
        x: card.x + 0.24,
        y: card.y + 0.28,
        w: card.w - 0.48,
        h: 0.14,
        fontSize: 9.2,
        bold: true,
        color: COLORS.white,
        align: "center",
      });
      let cy = card.y + 0.78;
      for (const line of card.lines) {
        slide.addShape(PptxGenJS.ShapeType.ellipse, {
          x: card.x + 0.26,
          y: cy + 0.08,
          w: 0.08,
          h: 0.08,
          fill: { color: card.accent },
          line: { color: card.accent },
        });
        addText(slide, line, {
          x: card.x + 0.4,
          y: cy,
          w: card.w - 0.58,
          h: 0.26,
          fontSize: 10,
          color: COLORS.ink,
        });
        cy += 0.56;
      }
    }
    addPanel(slide, 1.2, 5.7, 11.1, 0.72, COLORS.white, COLORS.line);
    addRichText(slide, [
      { text: "핵심 메시지  ", options: { bold: true, color: COLORS.navy } },
      { text: "큰 모델 하나가 답이 아니라 ", options: { color: COLORS.ink } },
      { text: "데이터 확대 + 안정적인 선택지 비교 + 편향 제거", options: { bold: true, color: COLORS.coral } },
      { text: "를 함께 묶었을 때 public score가 올라갔습니다.", options: { color: COLORS.ink } },
    ], {
      x: 1.5,
      y: 5.96,
      w: 10.55,
      h: 0.2,
      fontSize: 14,
      fontFace: FONT_HEAD,
      align: "center",
    });
    addAudit(slide, pptx);
  }

  // Slide 4
  {
    const slide = pptx.addSlide("MASTER_WIDE");
    addSlideFrame(slide, "03", "3B 베이스라인에서 7B 튜닝 전략으로", "하이퍼파라미터 변화는 각각 이유가 있었고, 모두 작은 시각 단서를 더 잘 잡기 위한 방향이었습니다.");
    addPanel(slide, 0.86, 2.08, 6.0, 4.98);
    addText(slide, "변경 요약", {
      x: 1.05,
      y: 2.34,
      w: 1.15,
      h: 0.18,
      fontSize: 13,
      fontFace: FONT_HEAD,
      bold: true,
    });
    addTag(slide, 2.68, 2.31, 1.02, "Baseline", COLORS.soft, COLORS.muted);
    addTag(slide, 4.28, 2.31, 1.32, "Final / A", COLORS.paleTeal, COLORS.teal);
    addComparisonRow(slide, 2.9, "모델", "3B", "7B", "", COLORS.teal);
    addComparisonRow(slide, 3.38, "해상도", "384px", "672px", "", COLORS.green);
    addComparisonRow(slide, 3.86, "LoRA rank", "8", "64", "", COLORS.green);
    addComparisonRow(slide, 4.34, "LR", "1e-4", "1e-5", "", COLORS.coral);
    addComparisonRow(slide, 4.82, "Epoch", "1", "3", "", COLORS.amber);
    addComparisonRow(slide, 5.3, "Scheduler", "linear", "cosine", "", COLORS.teal);
    addComparisonRow(slide, 5.78, "GPU", "RTX 5060 Ti", "H100 80GB", "", COLORS.teal);
    addPanel(slide, 7.05, 2.08, 5.45, 4.98, COLORS.white, COLORS.line);
    addText(slide, "왜 이 조합이 먹혔나", {
      x: 7.35,
      y: 2.34,
      w: 2.2,
      h: 0.18,
      fontSize: 13,
      fontFace: FONT_HEAD,
      bold: true,
    });
    addBulletBlock(slide, 7.35, 2.74, 4.75, "세 가지 핵심 가설", [
      "재활용 분류는 작은 시각 단서에 민감하므로 해상도와 모델 크기가 직접 성능으로 이어진다.",
      "LoRA rank를 키운 만큼 LR를 낮춰야 과도한 업데이트 없이 안정적으로 적응한다.",
      "1 epoch으로는 부족하고, cosine으로 후반부를 눌러줘야 미세한 경계가 정리된다.",
    ], COLORS.teal);
    addPanel(slide, 7.35, 5.42, 4.75, 1.16, COLORS.paleAmber, COLORS.paleAmber);
    addText(slide, "결과적으로 모델 capacity를 키우되, 학습은 더 조심스럽게 움직이는 방향으로 설계했습니다.", {
      x: 7.62,
      y: 5.73,
      w: 4.18,
      h: 0.36,
      fontSize: 10.8,
      fontFace: FONT_HEAD,
      bold: true,
      color: COLORS.navy,
      align: "center",
    });
    addAudit(slide, pptx);
  }

  // Slide 5
  {
    const slide = pptx.addSlide("MASTER_WIDE");
    addSlideFrame(slide, "04", "데이터 전략: FULL_DATA_MODE로 학습량 확대", "dev를 버리지 않고 다수결 gold로 흡수하면서, 학습 데이터 부족 문제를 가장 직접적으로 해결했습니다.");
    addPanel(slide, 0.88, 2.14, 4.15, 4.88);
    addText(slide, "데이터 볼륨 변화", {
      x: 1.12,
      y: 2.4,
      w: 2.0,
      h: 0.18,
      fontSize: 13,
      fontFace: FONT_HEAD,
      bold: true,
    });
    addText(slide, "train only", {
      x: 1.15,
      y: 3.0,
      w: 0.8,
      h: 0.16,
      fontSize: 10,
      bold: true,
    });
    addBar(slide, 1.15, 3.28, 2.95, 0.36, 5073 / 9486, COLORS.teal);
    addText(slide, "5,073", {
      x: 1.15,
      y: 3.72,
      w: 0.9,
      h: 0.16,
      fontSize: 15,
      fontFace: FONT_HEAD,
      bold: true,
      color: COLORS.teal,
    });
    addText(slide, "train + dev majority vote", {
      x: 1.15,
      y: 4.42,
      w: 1.9,
      h: 0.16,
      fontSize: 10,
      bold: true,
    });
    addBar(slide, 1.15, 4.7, 2.95, 0.36, 1.0, COLORS.green);
    addText(slide, "9,486", {
      x: 1.15,
      y: 5.14,
      w: 0.9,
      h: 0.16,
      fontSize: 15,
      fontFace: FONT_HEAD,
      bold: true,
      color: COLORS.green,
    });
    addText(slide, "+87% 데이터 확대", {
      x: 2.25,
      y: 5.14,
      w: 1.5,
      h: 0.16,
      fontSize: 11.5,
      fontFace: FONT_HEAD,
      bold: true,
      color: COLORS.navy,
    });
    addPanel(slide, 5.28, 2.14, 3.0, 4.88, COLORS.white, COLORS.line);
    addText(slide, "dev를 어떻게 썼나", {
      x: 5.56,
      y: 2.4,
      w: 1.7,
      h: 0.18,
      fontSize: 13,
      fontFace: FONT_HEAD,
      bold: true,
    });
    const answers = ["answer1", "answer2", "answer3", "answer4", "answer5"];
    let ay = 3.02;
    for (let i = 0; i < answers.length; i++) {
      slide.addShape(PptxGenJS.ShapeType.roundRect, {
        x: 5.62,
        y: ay,
        w: 1.0,
        h: 0.34,
        rectRadius: 0.08,
        fill: { color: i < 3 ? COLORS.paleGreen : COLORS.soft },
        line: { color: i < 3 ? COLORS.green : COLORS.line },
      });
      addText(slide, answers[i], {
        x: 5.62,
        y: ay + 0.08,
        w: 1.0,
        h: 0.12,
        fontSize: 8.5,
        align: "center",
        bold: true,
        color: i < 3 ? COLORS.green : COLORS.muted,
      });
      ay += 0.52;
    }
    slide.addShape(PptxGenJS.ShapeType.chevron, {
      x: 6.62,
      y: 4.0,
      w: 0.42,
      h: 0.52,
      fill: { color: COLORS.amber },
      line: { color: COLORS.amber },
      rotate: 90,
    });
    addPanel(slide, 7.32, 3.85, 0.82, 0.82, COLORS.paleAmber, COLORS.paleAmber);
    addText(slide, "gold", {
      x: 7.32,
      y: 4.12,
      w: 0.82,
      h: 0.14,
      fontSize: 11,
      bold: true,
      color: COLORS.amber,
      align: "center",
    });
    addPanel(slide, 8.48, 2.14, 4.0, 4.88, COLORS.white, COLORS.line);
    addText(slide, "운영 방식", {
      x: 8.74,
      y: 2.4,
      w: 1.2,
      h: 0.18,
      fontSize: 13,
      fontFace: FONT_HEAD,
      bold: true,
    });
    addBulletBlock(slide, 8.74, 2.76, 3.35, "학습 세팅", [
      "train + dev 전체를 학습에 사용",
      "검증은 100샘플만 모니터링 용도로 유지",
      "patience=999로 early stopping 사실상 비활성화",
      "3 epoch를 고정으로 끝까지 학습",
    ], COLORS.green);
    addText(slide, "데이터가 부족한 상황에서는 정교한 regularization보다 usable label을 더 확보하는 쪽이 효과적이었습니다.", {
      x: 8.74,
      y: 5.92,
      w: 3.3,
      h: 0.38,
      fontSize: 10.5,
      color: COLORS.navy,
      fontFace: FONT_HEAD,
      bold: true,
    });
    addAudit(slide, pptx);
  }

  // Slide 6
  {
    const slide = pptx.addSlide("MASTER_WIDE");
    addSlideFrame(slide, "05", "프롬프트는 짧지만 도메인 힌트는 분명하게", "정답 형식은 극단적으로 단순하게 유지하되, 재질/개수 구분에 필요한 관찰 포인트만 system prompt에 실었습니다.");
    addPanel(slide, 0.92, 2.16, 5.55, 4.86, COLORS.white, COLORS.line);
    addText(slide, "System prompt 핵심", {
      x: 1.2,
      y: 2.42,
      w: 1.9,
      h: 0.18,
      fontSize: 13,
      fontFace: FONT_HEAD,
      bold: true,
    });
    const promptBoxText = [
      "1. 개수 세기: 겹치거나 일부만 보여도 포함",
      "2. 재질 판별: 플라스틱 / 유리 / 금속 / 종이 / 비닐 / 스티로폼 구분",
      "3. 금속 vs 플라스틱: 광택, 반사, 찌그러짐 패턴으로 구분",
      "4. 출력 형식: a, b, c, d 중 하나의 소문자 한 글자",
    ];
    let py = 2.92;
    for (const line of promptBoxText) {
      addPanel(slide, 1.18, py, 4.95, 0.68, COLORS.soft, COLORS.soft);
      addText(slide, line, {
        x: 1.42,
        y: py + 0.2,
        w: 4.45,
        h: 0.24,
        fontSize: 10.2,
        color: COLORS.ink,
      });
      py += 0.88;
    }
    addPanel(slide, 6.72, 2.16, 5.63, 4.86, COLORS.white, COLORS.line);
    addText(slide, "왜 이렇게 설계했나", {
      x: 6.98,
      y: 2.42,
      w: 1.9,
      h: 0.18,
      fontSize: 13,
      fontFace: FONT_HEAD,
      bold: true,
    });
    addBulletBlock(slide, 6.98, 2.82, 4.85, "프롬프트 원칙", [
      "도메인 힌트는 넣되, 답안 형식은 한 글자로 강하게 제한",
      "객관식 보기는 user prompt에서 고정 포맷으로 전달",
      "금속/플라스틱처럼 자주 헷갈리는 경계 사례를 직접 언급",
      "복잡한 chain-of-thought보다 관찰 기준을 주는 쪽이 대회 환경에 더 실용적",
    ], COLORS.amber);
    addPanel(slide, 7.0, 5.58, 5.08, 0.94, COLORS.paleTeal, COLORS.paleTeal);
    addText(slide, "요약: 프롬프트는 모델을 '똑똑하게' 만들기보다, 시선을 어디에 둘지 정렬하는 장치로 사용했습니다.", {
      x: 7.28,
      y: 5.88,
      w: 4.56,
      h: 0.28,
      fontSize: 10.8,
      color: COLORS.navy,
      fontFace: FONT_HEAD,
      bold: true,
      align: "center",
    });
    addAudit(slide, pptx);
  }

  // Slide 7
  {
    const slide = pptx.addSlide("MASTER_WIDE");
    addSlideFrame(slide, "06", "generate 대신 logprob 직접 비교", "문자 하나를 뽑는 문제에서는 생성 문장을 후처리하는 것보다 a/b/c/d의 확률을 바로 비교하는 쪽이 훨씬 깨끗했습니다.");
    addPanel(slide, 0.88, 2.14, 5.66, 4.9, COLORS.white, COLORS.line);
    addText(slide, "기존 방식", {
      x: 1.18,
      y: 2.42,
      w: 1.0,
      h: 0.16,
      fontSize: 12,
      fontFace: FONT_HEAD,
      bold: true,
      color: COLORS.muted,
    });
    addPanel(slide, 1.18, 2.8, 5.05, 1.16, COLORS.soft, COLORS.soft);
    addText(slide, "model.generate() → 문자열 파싱 → a/b/c/d 추출", {
      x: 1.48,
      y: 3.14,
      w: 4.45,
      h: 0.2,
      fontSize: 13,
      fontFace: FONT_HEAD,
      bold: true,
      align: "center",
      color: COLORS.ink,
    });
    addBulletBlock(slide, 1.18, 4.36, 5.05, "문제점", [
      "불필요한 토큰이 섞이면 후처리 규칙이 복잡해짐",
      "선택지별 확신도를 직접 비교하기 어려움",
      "앙상블이나 TTA에서 점수를 평균내기 까다로움",
    ], COLORS.coral);
    addPanel(slide, 6.8, 2.14, 5.62, 4.9, COLORS.white, COLORS.line);
    addText(slide, "최종 방식", {
      x: 7.08,
      y: 2.42,
      w: 1.1,
      h: 0.16,
      fontSize: 12,
      fontFace: FONT_HEAD,
      bold: true,
      color: COLORS.teal,
    });
    addPanel(slide, 7.08, 2.8, 5.02, 1.36, COLORS.paleTeal, COLORS.paleTeal);
    addText(slide, "forward pass로 a / b / c / d 각 토큰의 logprob를 직접 추출", {
      x: 7.38,
      y: 3.08,
      w: 4.42,
      h: 0.42,
      fontSize: 12.5,
      fontFace: FONT_HEAD,
      bold: true,
      align: "center",
      color: COLORS.navy,
    });
    const probs = [
      ["a", -1.91, COLORS.soft],
      ["b", -0.73, COLORS.teal],
      ["c", -2.24, COLORS.soft],
      ["d", -1.35, COLORS.soft],
    ];
    let px = 7.35;
    for (const [label, value, fill] of probs) {
      addPanel(slide, px, 4.68, 1.0, 1.02, fill, fill);
      addText(slide, label, {
        x: px,
        y: 4.88,
        w: 1.0,
        h: 0.14,
        fontSize: 13,
        bold: true,
        align: "center",
        color: fill === COLORS.teal ? COLORS.white : COLORS.muted,
      });
      addText(slide, `${value}`, {
        x: px,
        y: 5.22,
        w: 1.0,
        h: 0.12,
        fontSize: 10,
        bold: true,
        align: "center",
        color: fill === COLORS.teal ? COLORS.white : COLORS.ink,
      });
      px += 1.16;
    }
    addText(slide, "장점: 확률 비교, 앙상블 평균, TTA 역매핑이 모두 같은 score space에서 처리됩니다.", {
      x: 7.08,
      y: 6.18,
      w: 5.02,
      h: 0.26,
      fontSize: 10.7,
      fontFace: FONT_HEAD,
      bold: true,
      color: COLORS.navy,
      align: "center",
    });
    addAudit(slide, pptx);
  }

  // Slide 8
  {
    const slide = pptx.addSlide("MASTER_WIDE");
    addSlideFrame(slide, "07", "TTA: 선택지 순서를 섞어 position bias 제거", "정답 그 자체보다 보기의 위치를 덜 믿게 만드는 것이 목적이었습니다.");
    addPanel(slide, 0.9, 2.16, 12.0, 4.9, COLORS.white, COLORS.line);
    addText(slide, "4개 permutation에서 같은 질문을 다시 묻고, logprob를 원래 위치로 역매핑한 뒤 평균합니다.", {
      x: 1.18,
      y: 2.42,
      w: 10.8,
      h: 0.2,
      fontSize: 12,
      color: COLORS.muted,
    });
    const chips = [
      { x: 1.18, y: 3.0, title: "Original", labels: ["a", "b", "c", "d"], accent: COLORS.teal },
      { x: 4.18, y: 3.0, title: "Shuffle 1", labels: ["b", "d", "a", "c"], accent: COLORS.green },
      { x: 7.18, y: 3.0, title: "Shuffle 2", labels: ["c", "a", "d", "b"], accent: COLORS.amber },
      { x: 10.18, y: 3.0, title: "Shuffle 3", labels: ["d", "c", "b", "a"], accent: COLORS.coral },
    ];
    for (const chip of chips) {
      addText(slide, chip.title, {
        x: chip.x + 0.54,
        y: chip.y - 0.28,
        w: 1.7,
        h: 0.16,
        fontSize: 10,
        bold: true,
        align: "center",
        color: chip.accent,
      });
      const svg = permutationChipSvg(chip.labels, chip.accent);
      slide.addImage({
        data: svg,
        ...imageSizingContain(svg, chip.x, chip.y, 2.55, 0.72),
      });
      slide.addShape(PptxGenJS.ShapeType.chevron, {
        x: chip.x + 0.95,
        y: chip.y + 1.04,
        w: 0.62,
        h: 0.36,
        fill: { color: chip.accent },
        line: { color: chip.accent },
        rotate: 90,
      });
      addPanel(slide, chip.x + 0.62, chip.y + 1.5, 1.32, 0.82, chip.accent, chip.accent);
      addText(slide, "logprob", {
        x: chip.x + 0.62,
        y: chip.y + 1.73,
        w: 1.32,
        h: 0.14,
        fontSize: 10.5,
        bold: true,
        align: "center",
        color: COLORS.white,
      });
    }
    slide.addShape(PptxGenJS.ShapeType.chevron, {
      x: 5.75,
      y: 5.38,
      w: 1.0,
      h: 0.42,
      fill: { color: COLORS.navy },
      line: { color: COLORS.navy },
    });
    addPanel(slide, 4.58, 5.96, 3.44, 0.72, COLORS.paleTeal, COLORS.paleTeal);
    addText(slide, "역매핑 후 평균 → 최종 argmax", {
      x: 4.58,
      y: 6.2,
      w: 3.44,
      h: 0.16,
      fontSize: 12,
      fontFace: FONT_HEAD,
      bold: true,
      align: "center",
      color: COLORS.navy,
    });
    addText(slide, "효과: 모델이 특정 위치의 선택지를 선호하더라도 평균 과정에서 편향이 상쇄됩니다.", {
      x: 1.18,
      y: 6.2,
      w: 2.6,
      h: 0.18,
      fontSize: 10.2,
      color: COLORS.ink,
    });
    addText(slide, "실전 결과: tuned Model A에서 TTA가 가장 높은 public score 0.91761을 기록했습니다.", {
      x: 8.45,
      y: 6.2,
      w: 3.72,
      h: 0.18,
      fontSize: 10.2,
      color: COLORS.ink,
      align: "right",
    });
    addAudit(slide, pptx);
  }

  // Slide 9
  {
    const slide = pptx.addSlide("MASTER_WIDE");
    addSlideFrame(slide, "08", "실험 결과: TTA가 최종 우승 전략", "성능 차이는 크지 않지만, 작은 차이를 만드는 요인이 무엇인지가 분명하게 드러났습니다.");
    addPanel(slide, 0.92, 2.12, 6.25, 4.94, COLORS.white, COLORS.line);
    addText(slide, "Public score 비교", {
      x: 1.2,
      y: 2.4,
      w: 1.6,
      h: 0.18,
      fontSize: 13,
      fontFace: FONT_HEAD,
      bold: true,
    });
    const resultBars = [
      { name: "7B + TTA (튜닝 전)", score: 0.91525, color: COLORS.amber, note: "기준선" },
      { name: "7B + 튜닝 + TTA (Model A)", score: 0.91761, color: COLORS.teal, note: "최고점" },
      { name: "7B + 앙상블 (A+B)", score: 0.91328, color: COLORS.coral, note: "하락" },
    ];
    const minScore = 0.912;
    const maxScore = 0.918;
    let ry = 3.08;
    for (const row of resultBars) {
      addText(slide, row.name, {
        x: 1.2,
        y: ry - 0.04,
        w: 2.15,
        h: 0.18,
        fontSize: 10,
        bold: true,
      });
      const fraction = (row.score - minScore) / (maxScore - minScore);
      addBar(slide, 3.52, ry, 2.8, 0.32, fraction, row.color, COLORS.soft);
      addText(slide, row.score.toFixed(5), {
        x: 3.72,
        y: ry + 0.4,
        w: 1.2,
        h: 0.16,
        fontSize: 13,
        fontFace: FONT_HEAD,
        bold: true,
        color: row.color,
      });
      addTag(slide, 5.18, ry + 0.42, 0.72, row.note, row.color, COLORS.white);
      ry += 1.18;
    }
    addPanel(slide, 1.18, 6.14, 5.66, 0.52, COLORS.paleGreen, COLORS.paleGreen);
    addText(slide, "+0.00236 improvement", {
      x: 1.18,
      y: 6.28,
      w: 5.66,
      h: 0.16,
      fontSize: 13.5,
      fontFace: FONT_HEAD,
      bold: true,
      color: COLORS.green,
      align: "center",
    });
    addPanel(slide, 7.42, 2.12, 5.08, 4.94, COLORS.white, COLORS.line);
    addText(slide, "해석", {
      x: 7.68,
      y: 2.4,
      w: 0.8,
      h: 0.18,
      fontSize: 13,
      fontFace: FONT_HEAD,
      bold: true,
    });
    addBulletBlock(slide, 7.68, 2.8, 4.3, "관찰 포인트", [
      "하이퍼파라미터 튜닝과 FULL_DATA_MODE가 기준선 대비 실질적인 상승을 만들었다.",
      "Model B를 따로 만든 것 자체는 나쁘지 않았지만, 앙상블이 항상 정답은 아니었다.",
      "작은 score 차이에서도 운영 전략이 최종 제출물을 바꾸는 결정적 요소가 됐다.",
    ], COLORS.teal);
    addPanel(slide, 7.68, 5.78, 4.3, 0.84, COLORS.paleAmber, COLORS.paleAmber);
    addText(slide, "대회형 문제에서는 '복잡한 방법'보다 '안정적으로 반복 가능한 방법'이 더 강했습니다.", {
      x: 7.94,
      y: 6.04,
      w: 3.78,
      h: 0.24,
      fontSize: 10.8,
      fontFace: FONT_HEAD,
      bold: true,
      color: COLORS.navy,
      align: "center",
    });
    addAudit(slide, pptx);
  }

  // Slide 10
  {
    const slide = pptx.addSlide("MASTER_WIDE");
    addSlideFrame(slide, "09", "왜 앙상블이 실패했는가, 그리고 다음 액션", "잘못된 자동선택은 validation 설계와 모델 다양성 부족이 겹쳤을 때 쉽게 발생합니다.");
    addPanel(slide, 0.95, 2.16, 7.55, 4.28, COLORS.white, COLORS.line);
    addText(slide, "앙상블 하락 원인", {
      x: 1.22,
      y: 2.42,
      w: 1.8,
      h: 0.18,
      fontSize: 13,
      fontFace: FONT_HEAD,
      bold: true,
    });
    const reasons = [
      {
        title: "validation overfit",
        body: "FULL_DATA_MODE에서 validation이 학습 데이터 일부 100샘플뿐이라, 앙상블 우위가 과적합된 숫자였음",
        fill: COLORS.paleCoral,
        accent: COLORS.coral,
      },
      {
        title: "lack of diversity",
        body: "A/B 모두 같은 아키텍처와 거의 같은 데이터로 학습되어 에러 패턴이 충분히 달라지지 않았음",
        fill: COLORS.paleAmber,
        accent: COLORS.amber,
      },
      {
        title: "selection rule mismatch",
        body: "val 기준 자동선택 규칙은 있었지만, public leaderboard 일반화 성능까지는 보장하지 못했음",
        fill: COLORS.paleTeal,
        accent: COLORS.teal,
      },
    ];
    let rx = 1.22;
    for (const reason of reasons) {
      addPanel(slide, rx, 3.0, 2.12, 2.96, reason.fill, reason.fill);
      addText(slide, reason.title, {
        x: rx + 0.18,
        y: 3.24,
        w: 1.76,
        h: 0.18,
        fontSize: 10.4,
        fontFace: FONT_HEAD,
        bold: true,
        color: reason.accent,
        align: "center",
      });
      addText(slide, reason.body, {
        x: rx + 0.18,
        y: 3.74,
        w: 1.76,
        h: 1.52,
        fontSize: 9.4,
        color: COLORS.ink,
        align: "center",
      });
      rx += 2.44;
    }
    addPanel(slide, 8.8, 2.16, 3.7, 2.62, COLORS.white, COLORS.line);
    addText(slide, "이번 대회에서 남은 교훈", {
      x: 9.1,
      y: 2.46,
      w: 3.1,
      h: 0.18,
      fontSize: 13,
      fontFace: FONT_HEAD,
      bold: true,
    });
    addSimpleBullets(slide, 9.1, 2.9, 3.0, [
      "앙상블은 다양성이 있을 때만 유효하다.",
      "full-data 학습에서는 validation 신뢰도 설계가 더 중요하다.",
      "position bias 제거처럼 직접적인 추론 보정이 의외로 강하다.",
    ], COLORS.green, 9.8);
    addPanel(slide, 8.8, 5.02, 3.7, 1.48, COLORS.white, COLORS.line);
    addText(slide, "다음에 해볼 것", {
      x: 9.1,
      y: 5.28,
      w: 1.6,
      h: 0.18,
      fontSize: 13,
      fontFace: FONT_HEAD,
      bold: true,
    });
    addSimpleBullets(slide, 9.1, 5.66, 3.0, [
      "다른 VLM과의 heterogeneous ensemble",
      "cross-validation + OCR / prompt ensemble",
    ], COLORS.coral, 9.4);
    addPanel(slide, 0.95, 6.56, 7.55, 0.46, COLORS.paleGreen, COLORS.paleGreen);
    addText(slide, "Final takeaway: 최고점은 큰 모델 하나보다 학습량 확대와 추론 통제를 묶은 운영 전략에서 나왔습니다.", {
      x: 1.15,
      y: 6.69,
      w: 7.15,
      h: 0.18,
      fontSize: 11.5,
      fontFace: FONT_HEAD,
      bold: true,
      color: COLORS.navy,
      align: "center",
    });
    addAudit(slide, pptx);
  }

  return pptx;
}

async function main() {
  ensureDir(OUT_DIR);
  const pptx = buildDeck();
  await pptx.writeFile({ fileName: OUT_FILE });
  console.log(`Wrote ${OUT_FILE}`);
}

module.exports = { buildDeck, OUT_FILE };

if (require.main === module) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
