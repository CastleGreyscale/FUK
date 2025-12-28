# FUK: Framework for Unified Kreation

**Local-first rendering pipeline for professional image and video generation workflows.**

---

## What is FUK?

FUK is a spite-driven alternative to cloud-first AI rendering tools that force you to work around their limitations. Built for professionals who need AI as a render engine, not a constraint.

**Core Philosophy:**
- Your project structure, not forced folder hierarchies
- Stable workflows over cosmetic UI changes
- Professional formats (EXR, proper bit depth) as standard
- Local control without cloud dependencies

---

## Features

### Current
- *Coming soon - project in early development*

### Planned
A left-to-right tabbed pipeline:
- **Generation** - Image/video generation with integrated prompt expansion
- **ControlNet** - External image workflows without forced copying
- **Post-processing** - Upscaling, style transfer, interpretation
- **Technical Outputs** - Depth maps, crypto mattes, render passes

---

## Installation

*Installation instructions coming soon*

**Requirements:**
- Python 3.10
- Local AI models (details TBD)

---

## Usage

*Usage examples coming soon*

---

## Development Roadmap

### Phase 1: Foundation
- [ ] Project structure setup
- [ ] Core dependency management (pyproject.toml)
- [ ] Basic Gradio UI framework
- [ ] Tab structure implementation

### Phase 2: Generation Pipeline
- [ ] Image generation (musubi-tuner integration)
- [ ] Metadata storage system (SQLite)
- [ ] Basic file I/O without forced copying

### Phase 3: ControlNet & External Images
- [ ] ControlNet preprocessor integration
- [ ] External image input handling
- [ ] Flexible file path management
- [ ] Batch processing support

### Phase 4: Post-Processing
- [ ] Upscaling integration
- [ ] Style transfer/interpretation tools
- [ ] Format conversion utilities
- [ ] EXR support with OpenImageIO

### Phase 5: Technical Outputs
- [ ] Depth map generation
- [ ] Crypto matte support
- [ ] Additional render passes
- [ ] Multi-format export

### Phase 6: Polish
- [ ] Performance optimization
- [ ] Comprehensive documentation
- [ ] Example workflows
- [ ] Memory management improvements
- [ ] Prompt expansion integration (Ollama + Qwen)

---

## Why FUK?

Built out of frustration with tools that:
- Force file structures that break professional workflows
- Require constant relaunches for basic settings
- Prioritize cloud deployment over local user experience
- Remove rollback capabilities in the name of "security"
- Lack support for professional formats (EXR, proper bit depth)

FUK does one thing: let you render without fighting your tools.

---

## Contributing

*Contribution guidelines coming soon*

---

## Credits

FUK uses [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) 
by kohya-ss for inference. Licensed under Apache 2.0.

---

## License

*License TBD*

---

## Name

Yes, it's called FUK. **F**ramework for **U**nified **K**reation.

Get FUKed. Render locally.
