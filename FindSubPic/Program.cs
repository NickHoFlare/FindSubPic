using OpenCvSharp;
using System.Diagnostics.Metrics;

// Supported Filetypes: https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
using (var t = new ResourcesTracker())
{
    var source = t.T(new Mat("./assets/multipic.jpg", ImreadModes.Unchanged));
    var grayscale = t.T(new Mat("./assets/multipic.jpg", ImreadModes.Grayscale));
    var edgeDetection = t.NewMat();
    var dilated = t.NewMat();

    // Detect edges
    edgeDetection = DetectEdges(grayscale, edgeDetection);

    // Apply dilation, then close it (This was the key to recognizing all 4!)
    dilated = DilateAndClose(edgeDetection, dilated);

    // Find contours
    var contours = FindContours(dilated);

    if (contours == null || contours.Length == 0)
    {
        Console.WriteLine("No contours found.");
        return;
    }

    // Find photos amongst the contours
    var photos = FindPhotos(source, contours, t);

    // Display the images
    Cv2.ImShow("source", source);
    Cv2.ImShow("edgeDetection", edgeDetection);
    Cv2.ImShow("dilated", dilated);
    DisplayPhotos(photos);

    Cv2.WaitKey();
}

Mat DetectEdges(Mat source, Mat detectEdgesInput)
{
    //Cv2.GaussianBlur(grayscale, grayscale, new Size(5, 5), 0.5);
    Cv2.Canny(source, detectEdgesInput, 50, 200);

    return detectEdgesInput;
}

Mat DilateAndClose(Mat source, Mat dilateAndCloseInput)
{
    Cv2.Dilate(source, dilateAndCloseInput, null);
    using (Mat kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3)))
    {
        Cv2.MorphologyEx(dilateAndCloseInput, dilateAndCloseInput, MorphTypes.Close, kernel);
    }

    return dilateAndCloseInput;
}

Point[][]? FindContours(Mat source)
{
    Cv2.FindContours(source, out Point[][]? contours, out HierarchyIndex[] hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
    contours = contours.Where(c => Cv2.ContourArea(c) > 1000).ToArray();

    return contours;
}

List<Mat> FindPhotos(Mat originalSource, Point[][] contours, ResourcesTracker t)
{
    var photos = new List<Mat>();

    var counter = 0;
    foreach (var contour in contours)
    {
        // Approximate the contour to a polygon
        var approx = Cv2.ApproxPolyDP(contour, 0.02 * Cv2.ArcLength(contour, true), true);

        // We want only rectangles
        if (approx.Length == 4 && Cv2.IsContourConvex(approx))
        {
            // Draw the contour (the outline of the rectangle)
            Cv2.DrawContours(originalSource, new Point[][] { approx }, 0, Scalar.Green, 2);

            // Crop the rectangle
            Rect boundingBox = Cv2.BoundingRect(approx);
            var photo = t.T(new Mat(originalSource, boundingBox));

            counter++;
            photos.Add(photo);
        }
    }

    return photos;
}

void DisplayPhotos(List<Mat> photos)
{
    for (var i = 0; i < photos.Count; i++)
    {
        Cv2.ImShow($"output{i}", photos[i]);
    }
}
