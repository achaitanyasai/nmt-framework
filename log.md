| Date           | Update | Commit id|Stable?|
|---|---|---|---|
| August 24 2019 | Base model which was used to build the future models by debugging and fixing version change issues | 2373da6772401afd2c712735fa2b0c180ee3101b | No
| ~~Sun Aug 25 22:18:19 IST 2019~~ | ~~Implemented Data Iterator and added tests testing it. Need to add more tests and test next_batch() function~~ | ~~c987086e9b2ac27f669b6dd4ebaca79314a1d2b1~~| No
| Sun Aug 25 22:52:30 IST 2019 | Implemented Data Iterator and added tests testing it. Check again for missed tests. <b>Fix the transpose issue of source and target tensors</b> | 72de01cc4b85785c1d4a0113f0f0708cfa159fda| No  
| Sat Nov 16 09:26:08 IST 2019 | Implemented all the remaining features and added tests | 3eef1ed70eca0cb2ddf9b05cfb1eb833b7d2bafb | No
| ~~Thu Dec 26 20:39:47 IST 2019~~ | ~~Added decoder output dependant attention. Also, implemented vaidate\_fixed method. Fixed the model.save method. Added tests except nmt.py and translator.py~~| ~~8f13a9d032c5578fee9ffbc33a46f8b762126ac1~~ | ~~No~~
|Mon Dec 30 01:18:12 IST 2019 | Optimized data iterator to use less memory. By default, the optimizer now uses patience instead of constant lrate decay. Modify in Optimizer.py if you need constant lrate decay|50e0709ce35264d5a2900aed0339e19b5eab3b05|**Yes**
