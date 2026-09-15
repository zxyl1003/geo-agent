# GeoExp7K-test Successful Examples

This directory contains five geographically diverse examples selected from the
current Gemma4-26B ExpGeoLoc evaluation on the GeoExp7K `test` split. All five
completed successfully with the released experience library and have
prediction-to-ground-truth distance at or below 1 km.

`samples.csv` records the original sample ID, image filename, ground-truth
location, model prediction, and Haversine error in meters. The images are copied
without resizing from the corresponding GeoExp7K-test records.

| Image | Ground truth | ExpGeoLoc prediction | Error |
| --- | --- | --- | ---: |
| <img src="images/img_c8acab0825003e1637c4fbac.jpg" width="360" alt="Pretoria example"> | Pretoria, South Africa | 625 President Steyn St, Pretoria | 0.00 m |
| <img src="images/img_bc2b593c947efc712641e96f.jpg" width="360" alt="Cerritos example"> | Cerritos, Mexico | Agencia de Viajes El Zaguán, Cerritos | 0.00 m |
| <img src="images/img_d01640ebf30106d7528ded5c.jpg" width="360" alt="Buritirana example"> | Buritirana, Brazil | Quadra Poliesportiva Davi Cantanhede | 0.00 m |
| <img src="images/img_a593d41507b5623556715947.jpg" width="360" alt="Innisfail example"> | Innisfail, Australia | North Coast Machinery, Innisfail | 0.00 m |
| <img src="images/img_780cd6a5292271dbf9fa8a34.jpg" width="360" alt="Piatra Neamț example"> | Piatra Neamț, Romania | Fly Music and CozlaVet, Piatra Neamț | 0.01 m |

The full dataset is available from
[Baidu Netdisk (code: `xbxr`)](https://pan.baidu.com/s/17wpiaDFg5ZiermPFtOBp5g?pwd=xbxr).
