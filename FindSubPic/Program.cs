using OpenCvSharp;

// Supported Filetypes: https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
using (var t = new ResourcesTracker())
{
    var source = t.T(new Mat("./assets/multipic.jpg", ImreadModes.Unchanged));
    var grayscale = t.T(new Mat("./assets/multipic.jpg", ImreadModes.Grayscale));
    var edgeDetection = t.NewMat();
    var dilated = t.NewMat();
    Mat output;

    // Detect edges
    //Cv2.GaussianBlur(grayscale, grayscale, new Size(5, 5), 0.5);
    Cv2.Canny(grayscale, edgeDetection, 50, 200);

    // Apply dilation (This was the key to recognizing all 4!)
    Cv2.Dilate(edgeDetection, dilated, null);

    // Find contours
    Cv2.FindContours(dilated, out Point[][]? contours, out HierarchyIndex[] hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
    contours = contours.Where(c => Cv2.ContourArea(c) > 1000).ToArray();


    var counter = 0;
    foreach (var contour in contours)
    {
        // Approximate the contour to a polygon
        var approx = Cv2.ApproxPolyDP(contour, 0.02 * Cv2.ArcLength(contour, true), true);

        // We want only rectangles
        if (approx.Length == 4 && Cv2.IsContourConvex(approx))
        {
            // Draw the contour (the outline of the rectangle)
            Cv2.DrawContours(source, new Point[][] { approx }, 0, Scalar.Green, 2);

            // Crop the rectangle
            Rect boundingBox = Cv2.BoundingRect(approx);
            output = t.T(new Mat(source, boundingBox));

            counter++;

            Cv2.ImShow($"output{counter}", output);
        }
    }

    // Display the images
    Cv2.ImShow("source", source);
    Cv2.ImShow("edgeDetection", edgeDetection);

    Cv2.WaitKey();
}
