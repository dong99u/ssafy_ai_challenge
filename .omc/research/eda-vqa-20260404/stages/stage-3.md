# Stage 3: Image Characteristics Analysis & IMAGE_SIZE Recommendation

**Date**: 2026-04-04
**Analyst**: Scientist Agent
**Status**: Complete

---

## Executive Summary

Analyzed 600 images (200 from each of train/dev/test splits) to determine optimal IMAGE_SIZE for Qwen2.5-VL fine-tuning.

**PRIMARY FINDING**: Recommend **IMAGE_SIZE = 672px** (vs current 384px)
- Preserves 62.5% of original image detail (vs 20.4% at 384px)
- 3.1x improvement in visual information retention
- Feasible within H100 80GB memory constraints
- Balanced trade-off between accuracy and computational cost

---

## Findings

### Finding F7: Image Resolution Statistics

#### Dataset Overview
- Total images analyzed: 600 (200 per split)
- Split consistency: High (mean differences < 1.2%)

#### Width Statistics (pixels)
| Metric | Train | Dev | Test | Mean |
|--------|-------|-----|------|------|
| Min | 720 | 720 | 719 | 720 |
| Q25 | 720 | 720 | 720 | 720 |
| Median | 720 | 720 | 720 | 720 |
| Mean | 775.5 | 766.6 | 766.5 | 770 |
| Q75 | 720 | 720 | 720 | 720 |
| Max | 1541 | 1525 | 1600 | 1600 |
| Std | 139.5 | 115.0 | 124.8 | 127 |

**Key insight**: 50% of images are exactly 720px wide; the remaining 50% range from 720–1600px.

#### Height Statistics (pixels)
| Metric | Train | Dev | Test | Mean |
|--------|-------|-----|------|------|
| Min | 720 | 720 | 720 | 720 |
| Q25 | 960 | 960 | 960 | 960 |
| Median | 960 | 960 | 960 | 960 |
| Mean | 937.6 | 933.9 | 946.5 | 939 |
| Q75 | 960 | 960 | 960 | 960 |
| Max | 1558 | 1565 | 1561 | 1565 |
| Std | 152.9 | 157.0 | 153.7 | 155 |

**Key insight**: Most images are either 720×720 or 720×960 (portrait). The typical aspect ratio is 0.75 (4:3).

#### Aspect Ratio Distribution
| Category | Count | % |
|----------|-------|-----|
| Portrait (< 0.9) | 475 | 79.2% |
| Square-ish (0.9–1.1) | 28 | 4.7% |
| Landscape (> 1.1) | 97 | 16.2% |

**Key insight**: Nearly 80% of images are portrait-oriented, requiring tall-aspect support.

#### File Size (KB)
| Metric | Train | Dev | Test | Mean |
|--------|-------|-----|------|------|
| Min | 34.8 | 46.2 | 48.7 | 43 |
| Median | 114.0 | 120.9 | 113.8 | 116 |
| Mean | 126.4 | 129.0 | 123.6 | 126 |
| Max | 387.1 | 287.3 | 286.2 | 387 |
| Std | 52.2 | 47.6 | 49.1 | 50 |

**Key insight**: Median ~120KB; images are well-compressed JPEGs.

---

### Finding F8: Optimal IMAGE_SIZE Recommendation

#### Detail Preservation Analysis

| IMAGE_SIZE | Pixel Info Retained | Downscaling % | Mem per Image (RGB) |
|------------|-------------------|--------------|--------------------|
| **384px** (current) | **20.4%** | 100% | 0.42 MB |
| **672px** (recommended) | **62.5%** | 100% | 1.29 MB |
| 1024px (alternative) | 145.1% | 3.5% | 3.00 MB |

**Statistical Evidence**:
- [STAT:n] Sample size: n = 600 images across three splits
- [STAT:consistency] Distribution consistency: width means differ by < 8.9px across splits (< 1.2% variation)
- [STAT:information_loss] At 384px: **100% of images lose detail** (downscaled from 720–1600px originals)
- [STAT:information_loss] At 672px: **62.5% of original visual information preserved** (3.1x improvement)

#### Computational Trade-offs

1. **672px Recommendation**:
   - Inference cost: **3.06x vs 384px** (quadratic scaling in image resolution)
   - Memory per image: 0.42MB → 1.29MB (3.07x increase)
   - **Feasible**: H100 80GB can support batch_size=1 with 672px + gradient accumulation
   - **Training speed**: Expect ~3x slower per batch; still acceptable for iterative fine-tuning

2. **1024px Alternative** (if accuracy is paramount):
   - Inference cost: 7.11x vs 384px
   - Memory per image: 3.00MB (demanding for batch_size > 1)
   - **Best detail preservation**: Only 3.5% of images upscaled (most native-size)
   - **Trade-off**: Slower training/inference; marginal accuracy gains beyond 672px

#### Why 672px Over 384px?

**Vision-Language Model Theory**:
- Qwen2.5-VL's vision encoder benefits from spatial detail
- Recycling item recognition (fine-grained classification) requires detail
- At 384px, all detail is lost; 672px recovers most without extreme cost

**Empirical Precedent**:
- LLaVA fine-tuning: defaults to 336–672px range
- CogVLM: uses 490–1024px range depending on task
- Industry standard: 672px is common "sweet spot" for VLM fine-tuning

**Aspect Ratio Suitability**:
- 79% portrait images → 672×672 square maintains vertical context better than 384×384
- 4:3 aspect (most common) maps naturally to 672×672 with padding

---

## Visualizations

Generated 4 figures in `.omc/scientist/figures/`:

1. **image_resolution_distribution.png**: Box plots of width/height across splits with reference lines for 384/672/1024px
2. **aspect_ratio_analysis.png**: Histogram and pie chart of aspect ratio categories
3. **width_height_scatter.png**: Scatter plot showing native resolutions vs proposed IMAGE_SIZE targets
4. **file_size_distribution.png**: Box plot of JPEG file sizes

---

## Limitations

1. **No empirical validation**: This recommendation is based on information theory and industry best practices, not controlled fine-tuning experiments. Actual accuracy improvement requires benchmark comparison (384px vs 672px vs 1024px).

2. **Sample variation**: Only 200 images per split; rare image types may have different statistics.

3. **Model-specific tuning**: Recommendation assumes Qwen2.5-VL benefits from detail preservation. Different architectures may have different optimal sizes.

4. **Memory model assumptions**: Estimates assume standard implementations; actual memory footprint depends on batch processing, dtype, and framework optimizations.

---

## Recommendations

### Immediate Action: Adopt 672px

```python
# In training config:
IMAGE_SIZE = 672
processor = AutoImageProcessor.from_pretrained(
    model_name,
    size={"height": 672, "width": 672},
    # ... other config
)
```

### Validation Plan
1. Train small pilot (1-2 epochs) with 384px (baseline) and 672px
2. Compare validation accuracy on dev.csv
3. If 672px shows improvement, proceed to full training
4. If marginal or no improvement, reconsider 1024px or stick with 384px

### Fallback
- If memory pressure: revert to 384px
- If accuracy insufficient: scale up to 1024px (requires reducing batch_size or gradient accumulation)

---

## Data Quality Notes

- No missing or corrupted images detected in sample
- All three splits (train/dev/test) have statistically similar distributions
- File sizes and aspect ratios are consistent across splits
- No evidence of outliers or data leakage

---

## References

- Qwen2.5-VL documentation: supports variable image sizes with tokenization
- LLaVA paper: image resolution impact on fine-tuning
- CogVLM paper: aspect ratio and detail preservation analysis
