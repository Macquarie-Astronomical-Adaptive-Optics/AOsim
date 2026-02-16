# Adaptive Optics simulator

(WIP) A User Interface / GUI for creating, editing, and running Adaptive Optics simulations


## Current features

System setup diagnostics:
- Poke matrix visualisation
- WFS editing
- Sub aperture centroids / images view
- Reference and Poke matrix Science image view
- FWHM, Diffraction Limit, Strehl Calculations
- Parameter config save/load

<img width="1708" height="1105" alt="image" src="https://github.com/user-attachments/assets/1020c745-bfbe-47e9-8f63-e90419ac3839" />

Turbulence 
- Turbulence phase screen generator/editor
- Layered turbulence viewer
  - Combined phase map view
  - Final PSF view

<img width="1708" height="1105" alt="image" src="https://github.com/user-attachments/assets/22c64c13-0bc7-40c8-8e31-0990e0b21901" />

Sensor & Turbulence Overview 
- Individual Sensor PSF grid
- Layer-by-layer view for WFS footprint/fov

<img width="1708" height="1105" alt="image" src="https://github.com/user-attachments/assets/252123dc-5056-4898-8046-1c269c007fab" />

Reconstructor view
- Reconstruction matrix visualization
- Phase screen -> reconstructed wavefront -> dm surface -> open-loop DM corrected residuals views

<img width="1708" height="1105" alt="image" src="https://github.com/user-attachments/assets/c9bc4534-4403-40eb-aa44-545e9cb3a12e" />

Closed-Loop simulation
- Real-time Statistics
  - PSF
  - Corrected & Uncorrected FWHM over time
  - Uncorrected r0 estimate
  - Wavefront Residuals
 
<img width="1708" height="1105" alt="image" src="https://github.com/user-attachments/assets/bfec8cb2-9a15-46b2-9ae3-fc248e35a0c8" />

Batched Long-exposure simulation
- Long-exposure PSF generation
  - PSF generation for N amount of off-axis points
  - FWHM estimation via Gaussian/Moffat/Contour fitting
  - PSF in FITS file format

 <img width="2560" height="1540" alt="image" src="https://github.com/user-attachments/assets/5235d9dc-5a7d-4d41-ae88-67d5f95dea26" />

