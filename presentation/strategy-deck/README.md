# SSAFY AI Challenge Strategy Deck

## Files

- `ssafy-ai-challenge-strategy.pptx`: 최종 발표자료
- `ssafy-ai-challenge-strategy.pdf`: PowerPoint 렌더 기준 PDF
- `assets/montage.png`: 전체 슬라이드 미리보기
- `assets/cover-hero-v1.png`: 표지용 raster cover asset
- `source/deck.js`: PptxGenJS 원본
- `source/generate_cover_asset.py`: 표지 asset 생성 스크립트
- `source/package.json`, `source/package-lock.json`: deck 빌드용 Node 의존성
- `source/pptxgenjs_helpers/`: slides skill helper 복사본

## Rebuild

```bash
cd presentation/strategy-deck/source
npm install
npm run build
```

재생성 결과물은 `presentation/strategy-deck/source/dist/` 아래에 생성된다.
원본 deck는 작업용 디렉터리에서 먼저 생성한 뒤 최종 산출물만 이 폴더로 복사했다.
