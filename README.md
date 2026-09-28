# Retinal OPL

This project emulates the retinal Outer Plexiform Layer (OPL) as a computer vision pipeline:

```
luminance -> log compression -> Difference of Gaussians (centre-surround) -> ON/OFF rectification
```

A second version adds specular-reflection suppression so mirror-finish surfaces don't produce false edges, plus contour-area filtering to discard small noise fragments. A low-light (scotopic) mode further suppresses noise-driven false edges in dim conditions.

## Structure

- `phase_1/` - the original algorithm (v1), run against static images.
- `no_reflection/` - the corrected algorithm (v2): specular suppression + contour area filtering, run against static images.
- `live_feed/` + `index.html` - a live webcam version of v2, running in the browser via OpenCV.js.

## Algorithm

1. **Luminance** - convert BGR to luminance using CIE 1931 photopic weights (0.299R + 0.587G + 0.114B).
2. **Log compression** - Weber-Fechner receptor compression, `s = c * log(1 + r)`.
3. **Difference of Gaussians (DoG)** - centre Gaussian (sigma_c = 1.5) minus surround Gaussian (sigma_s = 6.0), modelling a retinal ganglion cell's centre-surround receptive field.
4. **ON/OFF rectification** - positive DoG response is the ON channel, negative DoG response is the OFF channel.

**v2 additions** (`no_reflection/`, `live_feed/`):

- **Specular suppression** - pixels with low saturation and high value (HSV) are flagged as specular highlights and replaced with their local surround average before the DoG step.
- **Contour area filtering** - after the DoG, small edge fragments (area below a threshold) are discarded, keeping only edges large enough to be real object boundaries.
- **Scotopic (low-light) handling** (`live_feed/` only) - in dim frames, a small pre-blur pools noise before log compression, the DoG edge threshold adapts to the frame's own measured noise level, and frame-to-frame temporal blending is smoothed further.

## Running it

**Static image pipelines** (`phase_1/`, `no_reflection/`):

```
python3 retinal_opl.py <input_image_path>       # v1, from phase_1/
python3 retinal_opl.py <input_image_path>       # v2 without specular handling, from no_reflection/
python3 retinal_opl_v2.py <input_image_path>    # v2 with specular handling, from no_reflection/
```

Outputs are written next to the input image, in `opl_output/` or `opl_v2_output/`.

**Live webcam feed**:

Serve the project root over HTTP (a browser will not run this from a `file://` URL) and open `index.html`:

```
python3 -m http.server 8000
```

Then visit `http://localhost:8000/index.html`, allow camera access, and press Start. Sliders control the photoreceptor time constant (tau), gain, and focus.
