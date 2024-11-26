using FindSubPicUtil;

var subPics = new FindSubPic("./assets/multipic.jpg");

//subPics.DisplaySourceImage();
//subPics.DisplaySubPics();
subPics.SaveSubPics("./test", "test", Constants.ImageFileType.Jpg, true);
