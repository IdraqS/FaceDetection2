Hello,

Yes this isn't the proper way to do read me!!

But please if you could just check if the script runs. Pls change the model directory at the top accordingly. 

Seem to be having issues with closeness to webcam basically. The closer to the webcam the easier it can distinguish. Further away it recognised my father as me. Could be two reasons. I have trained the dataset on (75,75,1) input shape. Closer to camera = larger image
that gets downsized to input shape. Further away = must stretch the image out = less pixel density so it just becomes a 'recognising ANY face' exercise for the model.

But just to be sure that it's not because our facial features are similar, if you could test it detects you as not idraq that would be great. Please play around with going closer/further away from webcam. It is also limited by angles - something due to dataset size.
Model is trained on mostly face on angles of celebrities for 'not me' dataset - so not much to do there just an accepted limitation of the study.

Please let me know thoughts

Will upload the rest of the excel sheets, trained model scripts, k fold script, etc etc later. 

