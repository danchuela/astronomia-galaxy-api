# notebooks

Space for exploratory analysis and prototyping. Not part of the production pipeline.

Notebooks here can use `galaxy_core` directly — no API needed:

```python
from packages.galaxy_core.analyzer import BasicGalaxyAnalyzer, create_synthetic_image

analyzer = BasicGalaxyAnalyzer()
image = create_synthetic_image((128, 128))
seg = analyzer.segment_galaxy(image)
measurements = analyzer.measure_basic(image, seg)
```
