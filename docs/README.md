# Ryzen AI Software Documentation

Documentation site for the AMD Ryzen AI Software Platform, living in the `docs/` folder of the RyzenAI-SW repository.

Content was migrated from the previous Sphinx/RST site (https://ryzenai.docs.amd.com/en/latest/), reorganized into a simplified category structure, and extended with a GPU/Radeon section and the RyzenAI-SW examples.

## Local preview

```bash
npx mint dev
```

Then open http://localhost:3000.

## Structure

- `docs.json` - site configuration (navigation, theme, branding)
- `index.mdx` - landing page
- Category folders (one level each): `getting-started/`, `vision/`, `llms/`, `audio/`, `gpu-radeon/`, `windows-ml/`, `tools/`, `reference/`
- `images/` - inline diagrams
- `assets/` - site assets (favicon)

## Page ownership

Every page carries a hidden owner header, for example:

```
{/* owner: dwithchenna */}
```

CI uses this header to route failures to the responsible owner via GitHub @mention. Default owner is `@dwithchenna`. See `.github/scripts/generate_codeowners.py`.
