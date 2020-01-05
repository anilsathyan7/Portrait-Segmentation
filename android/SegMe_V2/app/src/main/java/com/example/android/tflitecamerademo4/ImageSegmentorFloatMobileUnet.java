/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.example.android.tflitecamerademo4;

import android.app.Activity;
import android.graphics.Bitmap;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_32FC3;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.core.CvType.CV_8UC3;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2Lab;
import static org.opencv.imgproc.Imgproc.COLOR_BGRA2BGR;
import static org.opencv.imgproc.Imgproc.COLOR_BGRA2RGB;
import static org.opencv.imgproc.Imgproc.COLOR_Lab2BGR;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2BGR;

/** This segmentor works with the float mobile-unet model. */
public class ImageSegmentorFloatMobileUnet extends ImageSegmentor {

  /** The mobile net requires additional normalization of the used input. */
  private static final float IMAGE_MEAN = 0;//127.5f;

  private static final float IMAGE_STD = 255.f;//127.5f;

  /**
   * An array to hold inference results, to be feed into Tensorflow Lite as outputs. This isn't part
   * of the super class, because we need a primitive array here.
   */
  private float[][] segmap = null;

  int opsize=128;
  int k=0;
  int frame_hieght=0;
  int frame_width=0;

  Bitmap resbmp, tmpbmp,harbmp ;
  Boolean prop_set = Boolean.FALSE;


  /**
   * Initializes an {@code ImageSegmentorFloatMobileUnet}.
   *
   * @param activity
   */
  ImageSegmentorFloatMobileUnet(Activity activity) throws IOException {
    super(activity);
      segmap = new float[1][opsize*opsize];

  }

  @Override
  protected String getModelPath() {

    return "deconv_fin_munet.tflite";

  }


  @Override
  protected int getImageSizeX() {
    return 128;
  }

  @Override
  protected int getImageSizeY() {
    return 128;
  }

  @Override
  protected int getNumBytesPerChannel() {
    return 4; // Float.SIZE / Byte.SIZE;
  }

  @Override
  protected void addPixelValue(int pixelValue) {
    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN)/ IMAGE_STD);
    imgData.putFloat((((pixelValue >> 8) & 0xFF)  - IMAGE_MEAN)/ IMAGE_STD);
    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN)/ IMAGE_STD);
  }

  @Override
  protected void runInference( ) {

    tflite.run(imgData, segmap);
  }

  protected void set_frame_props(){

      /* Set size of global frames and initialize bitmaps */
      frame_hieght=Camera2BasicFragment.textureView.getHeight();
      frame_width=Camera2BasicFragment.textureView.getWidth();
      resbmp = Bitmap.createBitmap(frame_width,frame_hieght, Bitmap.Config.ARGB_8888);
      tmpbmp = Bitmap.createBitmap(frame_width,frame_hieght, Bitmap.Config.ARGB_8888);
      harbmp = Bitmap.createBitmap(512,512, Bitmap.Config.ARGB_8888);

  }

  @Override
  public Bitmap imageblend(Bitmap fg_bmp, Bitmap bg_bmp, Boolean harmonize){

    //Set global frame sizes & bitmaps on init()
    if (!prop_set) {
        set_frame_props();
        prop_set=Boolean.TRUE;
    }

    Mat fg = new Mat(frame_hieght,frame_width,CV_8UC3);
    Mat bg = new Mat(frame_hieght,frame_width,CV_8UC3);
    Mat bgcpy = new Mat(frame_hieght,frame_width,CV_8UC3);

    Mat mskmat = new Mat (opsize, opsize, CV_32F);
    Mat invmskmat = new Mat(frame_hieght,frame_width,CV_32FC3, new Scalar(1.0,1.0,1.0));

    Mat resmat = new Mat(frame_hieght,frame_width,CV_32FC3);

    if (segmap!=null){

      mskmat.put(0,0,segmap[0]);

      Imgproc.threshold(mskmat,mskmat,Camera2BasicFragment.mskthresh/100.0,1.0,Imgproc.THRESH_TOZERO);

      Imgproc.GaussianBlur(mskmat,mskmat,new Size(7,7),0);
      Core.pow(mskmat,2.0,mskmat);

      Imgproc.resize(mskmat,mskmat, new Size(frame_width, frame_hieght));
      Imgproc.cvtColor(mskmat,mskmat,Imgproc.COLOR_GRAY2BGR);

      Utils.bitmapToMat(fg_bmp, fg);
      Utils.bitmapToMat(bg_bmp, bg);

      Imgproc.cvtColor(fg,fg, COLOR_BGRA2BGR);
      Imgproc.cvtColor(bg,bg, COLOR_BGRA2BGR);

      if(harmonize) {
        // Deep copy the background mat
        bg.copyTo(bgcpy);
        fg = color_transfer(bgcpy, fg);
      }

      Log.d("FG", String.valueOf(fg.type()));
      Log.d("BG", String.valueOf(bg.type()));

      fg.convertTo(fg,CV_32FC3,1.0/255.0);
      bg.convertTo(bg,CV_32FC3,1.0/255.0);

      Core.subtract(invmskmat,mskmat,invmskmat);
      Core.multiply(fg,mskmat,fg);
      Core.multiply(bg,invmskmat, bg);
      Core.add(fg,bg,resmat);

      resmat.convertTo(resmat,CV_8UC3,255);

      Utils.matToBitmap(resmat,resbmp);

    }
    return resbmp;
  }


  @Override
  public Bitmap videobokeh(Bitmap fg_bmp){

    Mat fg = new Mat(frame_hieght,frame_width,CV_8UC3);
    Mat bg = new Mat(frame_hieght,frame_width,CV_8UC3);
    Mat mskmat = new Mat (opsize, opsize, CV_32F);
    Mat invmskmat = new Mat(frame_hieght,frame_width,CV_32FC3, new Scalar(1.0,1.0,1.0));

    Mat resmat = new Mat(frame_hieght,frame_width,CV_32FC3);

    if (segmap!=null){

      mskmat.put(0,0,segmap[0]);

      // Prepare the mask
      Core.multiply(mskmat,new Scalar(2.0),mskmat);
      Imgproc.threshold(mskmat, mskmat,1.0,1.0,Imgproc.THRESH_TRUNC);
      Core.pow(mskmat,2.0,mskmat);

      Imgproc.resize(mskmat,mskmat, new Size(frame_width, frame_hieght));
      Imgproc.cvtColor(mskmat,mskmat,Imgproc.COLOR_GRAY2BGR);

      Utils.bitmapToMat(fg_bmp, fg);
      Utils.bitmapToMat(fg_bmp, bg);
      Imgproc.resize(bg,bg, new Size(frame_width, frame_hieght));
      Imgproc.resize(fg,fg, new Size(frame_width, frame_hieght));
      Imgproc.cvtColor(fg,fg, COLOR_BGRA2BGR);
      Imgproc.cvtColor(bg,bg, COLOR_BGRA2BGR);

      // Blur the mask
      Imgproc.resize(bg,bg, new Size(224,224));
      Imgproc.blur(bg,bg,new Size(11,11));
      Imgproc.resize(bg,bg, new Size(frame_width, frame_hieght));

      fg.convertTo(fg,CV_32FC3,1.0/255.0);
      bg.convertTo(bg,CV_32FC3,1.0/255.0);

      // Alpha blend fg with bg, using the mask
      Core.subtract(invmskmat,mskmat,invmskmat);
      Core.multiply(fg,mskmat,fg);
      Core.multiply(bg,invmskmat, bg);
      Core.add(fg,bg,resmat);

      resmat.convertTo(resmat,CV_8UC3,255);

      Utils.matToBitmap(resmat,resbmp);

    }
    return resbmp;
  }

  public Mat color_transfer(Mat src, Mat tgt) {


    long startTimeb = SystemClock.uptimeMillis();

    // Downsample the image
    Imgproc.resize(src,src, new Size(448,448));
    Imgproc.resize(tgt,tgt, new Size(448,448));

    Imgproc.cvtColor(src,src,COLOR_BGR2Lab,CV_32FC3);
    Imgproc.cvtColor(tgt,tgt,COLOR_BGR2Lab,CV_32FC3);


    //compute color statistics for the source image
    List<Mat> labsrc = new ArrayList<Mat>(3);
    Core.split(src, labsrc);


    MatOfDouble meansrc= new MatOfDouble();
    MatOfDouble stdsrc= new MatOfDouble();

    Core.meanStdDev(src, meansrc, stdsrc);

    Log.d("meansrc",meansrc.dump() );

    Log.d("meanval1", String.valueOf(meansrc.get(0,0)[0]));
    Log.d("meanval2", String.valueOf(meansrc.get(1,0)[0]));
    Log.d("meanval3", String.valueOf(meansrc.get(2,0)[0]));

    double lMeanSrc = meansrc.get(0,0)[0];
    double aMeanSrc = meansrc.get(1,0)[0];
    double bMeanSrc = meansrc.get(2,0)[0];

    double lStdSrc = stdsrc.get(0,0)[0];
    double aStdSrc = stdsrc.get(1,0)[0];
    double bStdSrc = stdsrc.get(2,0)[0];


    //compute color statistics for the target image
    List<Mat> labtgt = new ArrayList<Mat>(3);
    Core.split(tgt, labtgt);

    MatOfDouble meantgt = new MatOfDouble();
    MatOfDouble stdtgt = new MatOfDouble();

    Core.meanStdDev(src, meantgt, stdtgt);

    double lMeanTgt = meantgt.get(0,0)[0];
    double aMeanTgt = meantgt.get(1,0)[0];
    double bMeanTgt = meantgt.get(2,0)[0];


    double lStdTgt = stdtgt.get(0,0)[0];
    double aStdTgt = stdtgt.get(1,0)[0];
    double bStdTgt = stdtgt.get(2,0)[0];


    // subtract the means from the target image
    Core.subtract(labtgt.get(0),new Scalar(lMeanTgt),labtgt.get(0));
    Core.subtract(labtgt.get(1),new Scalar(aMeanTgt),labtgt.get(1));
    Core.subtract(labtgt.get(2),new Scalar(bMeanTgt),labtgt.get(2));

    // scale by the standard deviations
    Core.multiply(labtgt.get(0),new Scalar((lStdTgt/lStdSrc)),labtgt.get(0));
    Core.multiply(labtgt.get(1),new Scalar((aStdTgt/aStdSrc)),labtgt.get(1));
    Core.multiply(labtgt.get(2),new Scalar((bStdTgt/bStdSrc)),labtgt.get(2));

    // add in the source mean
    Core.add(labtgt.get(0),new Scalar(lMeanSrc),labtgt.get(0));
    Core.add(labtgt.get(1),new Scalar(aMeanSrc),labtgt.get(1));
    Core.add(labtgt.get(2),new Scalar(bMeanSrc),labtgt.get(2));

    Core.merge(labtgt,tgt);
    Log.d("tgt_size", tgt.size().toString());
    Log.d("tgt_type", String.valueOf(tgt.type()));

    Imgproc.cvtColor(tgt,tgt,COLOR_Lab2BGR);
    tgt.convertTo(tgt,CV_8UC3);

    //Upsample the image
    Imgproc.resize(tgt,tgt, new Size(frame_width,frame_hieght));

    long endTimeb = SystemClock.uptimeMillis();

    Log.d("color transfer", Long.toString(endTimeb - startTimeb));

    return tgt;
  }


    public Bitmap smoothblend(Bitmap fg_bmp, Bitmap bg_bmp, Boolean harmozize){

    Mat mskmat = new Mat (opsize, opsize, CV_32F);

    if (segmap!=null){

      //Binarize the mask
      mskmat.put(0,0,segmap[0]);
      Imgproc.threshold(mskmat,mskmat,0.5,1.0,Imgproc.THRESH_BINARY);

      //Gaussian blur
      mskmat.convertTo(mskmat,CV_8UC3,255);
      Imgproc.GaussianBlur(mskmat,mskmat,new Size(9,9),0);

      // Resize the mask
      Imgproc.resize(mskmat,mskmat, new Size(frame_width,frame_hieght));

      //Apply smooth and alpha blend filters
      Utils.matToBitmap(mskmat,tmpbmp);
      resbmp=Camera2BasicFragment.renderSmooth(bg_bmp,fg_bmp,tmpbmp);

    }
    return resbmp;
  }

  @Override
  public void color_harmonize(){

      //Get mask from previous stage
      Mat mskmat = new Mat (opsize, opsize, CV_32F);
      mskmat.put(0,0,segmap[0]);
      Imgproc.resize(mskmat, mskmat, new Size(512,512));
      Imgproc.threshold(mskmat,mskmat,0.5,1.0,Imgproc.THRESH_BINARY);
      mskmat.convertTo(mskmat,CV_32FC3,255.0);
      Core.subtract(mskmat,new Scalar(128.0),mskmat);

      // Get output from previous stage
      Mat srcmat = new Mat();
      Utils.bitmapToMat(resbmp,srcmat);
      Imgproc.resize(srcmat, srcmat, new Size(512,512));
      Imgproc.cvtColor(srcmat,srcmat, COLOR_BGRA2RGB);

      //Set the net inputs
      Mat blobimg = Dnn.blobFromImage(srcmat, 1,
              new Size(512, 512),
              new Scalar(104.00699, 116.66877, 122.67892), false, false);

      Mat blobmsk = Dnn.blobFromImage(mskmat, 1,
              new Size(512, 512));

      Camera2BasicFragment.net.setInput(blobimg,"data");
      Camera2BasicFragment.net.setInput(blobmsk,"mask");

      // Perform inference on inputs
      long startTimeb = SystemClock.uptimeMillis();
      Mat result=Camera2BasicFragment.net.forward();
      long endTimeb = SystemClock.uptimeMillis();

      Log.d("color harmonize", Long.toString(endTimeb - startTimeb));

      // Extract rgb image from output blob
      int H = result.size(2);
      int W = result.size(3);

      // Reshape it to a long vertical strip:
      Mat strip = result.reshape(1, H * 3);

      // Collect the color planes into a channels list:
      List<Mat> channels = new ArrayList<>();
      channels.add(strip.submat(0,H, 0,W));
      channels.add(strip.submat(H,2*H, 0,W));
      channels.add(strip.submat(2*H,3*H, 0,W));

      // Merge planes into final rgb image
      Mat rgb = new Mat();
      Core.merge(channels, rgb);

      // Add the mean value
      Core.add(rgb,new Scalar(104.00699, 116.66877, 122.67892),rgb);

      // Convert mat to bitmap
      Imgproc.cvtColor(rgb,rgb,COLOR_RGB2BGR);
      rgb.convertTo(rgb,CV_8UC3);
      Utils.matToBitmap(rgb,harbmp);

    // Save bitmap output
      save(resbmp,"Seg_normalized");
      save(harbmp,"Seg_harmonized");

  }

    public void save (Bitmap bmp, String name){


        String extStorageDirectory = Environment.getExternalStorageDirectory().toString();
        OutputStream outStream = null;
        File file = new File(extStorageDirectory, name+k+".png");
        k++;
        try {
            outStream = new FileOutputStream(file);
            bmp.compress(Bitmap.CompressFormat.PNG, 100, outStream);
            outStream.flush();
            outStream.close();
        } catch(Exception e) {

        }

    }

}
