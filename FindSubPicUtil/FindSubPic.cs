using FindSubPicUtil.Exceptions;
using OpenCvSharp;
using static FindSubPicUtil.Constants;

namespace FindSubPicUtil;

public class FindSubPic : IFindSubPic, IDisposable
{
    private ResourcesTracker _tracker;
    private Mat _source;
    private List<Mat> _subPics;
    
    public FindSubPic(string filePath)
    {
        _tracker = new ResourcesTracker();
        _source = _tracker.T(new Mat(filePath, ImreadModes.Unchanged));
        _subPics = ProcessSourceImage(_source, filePath);
    }

    private List<Mat> ProcessSourceImage(Mat source, string filePath)
    {
        var grayscale = _tracker.T(new Mat(filePath, ImreadModes.Grayscale));

        // Detect edges
        var edgeDetection = DetectEdges(grayscale);

        // Apply dilation, then close it (This was the key to recognizing all 4!)
        var dilated = DilateAndClose(edgeDetection);

        // Find contours
        var contours = FindContours(dilated) ?? Array.Empty<Point[]>();

        if (contours ==  null || !contours.Any())
        {
            throw new NoContoursException("No contours could be found, likely no subpics present in source image");
        }

        // Find photos amongst the contours
        return FindSubPics(source, contours);
    }

    private Mat DetectEdges(Mat source)
    {
        var detectEdges = _tracker.NewMat();

        //Cv2.GaussianBlur(grayscale, grayscale, new Size(5, 5), 0.5);
        Cv2.Canny(source, detectEdges, 50, 200);

        return detectEdges;
    }

    private Mat DilateAndClose(Mat source)
    {
        var dilated = _tracker.NewMat();

        Cv2.Dilate(source, dilated, null);
        using (Mat kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3)))
        {
            Cv2.MorphologyEx(dilated, dilated, MorphTypes.Close, kernel);
        }

        return dilated;
    }

    private Point[][]? FindContours(Mat source)
    {
        Cv2.FindContours(source, out Point[][]? contours, out HierarchyIndex[] hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
        contours = contours.Where(c => Cv2.ContourArea(c) > 1000).ToArray();

        return contours;
    }

    private List<Mat> FindSubPics(Mat originalSource, Point[][] contours)
    {
        var subPics = new List<Mat>();

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
                var photo = _tracker.T(new Mat(originalSource, boundingBox));

                counter++;
                subPics.Add(photo);
            }
        }

        return subPics;
    }

    public void DisplaySourceImage()
    {
        Cv2.ImShow("Source", _source);
        Cv2.WaitKey();
    }

    public void DisplaySubPics()
    {
        for (var i = 0; i < _subPics.Count; i++)
        {
            Cv2.ImShow($"SubPic{i+1}", _subPics[i]);
        }
        Cv2.WaitKey();
    }

    // TODO: Give each subpic unique filename
    public void SaveSubPics(string destinationDirectory, string fileName, ImageFileType fileType)
    {
        if (!Directory.Exists(destinationDirectory))
        {
            throw new InvalidDirectoryException("The destination directory does not exist.");
        }

        var destinationPath = $"{destinationDirectory}/{fileName}.{fileType.ToString().ToLower()}";
        
        foreach (var subPic in _subPics)
        {
            subPic.SaveImage(destinationPath);
        }
    }

    public void Dispose()
    {
        _tracker.Dispose();
    }
}

public interface IFindSubPic
{
    void DisplaySourceImage();
    void DisplaySubPics();
    void SaveSubPics(string destinationDirectory, string fileName, ImageFileType fileType);
}
