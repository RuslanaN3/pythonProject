# Parking Assistant

###  Prediction API part of parking assistant application.

In the project directory, you can run:

```
SET FLASK_APP=main.py
flask run
```

Available methods:

`POST /api/predict`

The response is predicted parking slots statuses from provided image.

To test api you can send image from camera 5 with various weather conditions and occupancy of parking lot.
Images are presented [here].

CNRPark+EXT is a dataset for visual occupancy detection of parking lots of roughly 150,000 labeled images 
(patches) of vacant and occupied parking spaces, built on a parking lot of 164 parking spaces.

[here]: http://cnrpark.it 