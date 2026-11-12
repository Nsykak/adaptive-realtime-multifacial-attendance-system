# Preparation changes

- Removed invisible zero-width characters that prevented parsing.
- Kept valid reference names aligned with successfully encoded faces.
- Skipped unreadable files and created the reference-image directory when missing.
- Wrote consistent three-column CSV rows and deduplicated by person and date.
- Added camera-open checks and guaranteed camera cleanup after loop errors.
- Used unannotated frames for registration, limited label characters, prevented overwrites and removed images that failed encoding.
- Added setup documentation and ignored local personal data.

The original upload remains unchanged. This prepared version needs local integration testing.
