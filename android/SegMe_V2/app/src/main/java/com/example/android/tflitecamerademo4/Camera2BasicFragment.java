/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
import android.app.AlertDialog;
import android.app.Dialog;
import android.app.DialogFragment;
import android.app.Fragment;
import android.content.Context;
import android.content.DialogInterface;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.PorterDuff;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.ImageReader;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.support.annotation.NonNull;
import android.support.v4.content.ContextCompat;
import android.text.SpannableString;
import android.text.SpannableStringBuilder;
import android.util.Log;
import android.util.Size;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.NumberPicker;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;
import android.support.v13.app.FragmentCompat;


import com.example.android.tflitecamerademo.ScriptC_saturation;

import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

import jp.co.cyberagent.android.gpuimage.GPUImage;
import jp.co.cyberagent.android.gpuimage.GPUImageView;
import jp.co.cyberagent.android.gpuimage.filter.GPUImageContrastFilter;
import jp.co.cyberagent.android.gpuimage.filter.GPUImageEmbossFilter;
import jp.co.cyberagent.android.gpuimage.filter.GPUImageFilter;
import jp.co.cyberagent.android.gpuimage.filter.GPUImageFilterGroup;
import jp.co.cyberagent.android.gpuimage.filter.GPUImageKuwaharaFilter;
import jp.co.cyberagent.android.gpuimage.filter.GPUImageSepiaToneFilter;
import jp.co.cyberagent.android.gpuimage.filter.GPUImageToneCurveFilter;


/** Basic fragments for the Camera. */
public class Camera2BasicFragment extends Fragment
    implements FragmentCompat.OnRequestPermissionsResultCallback {

  /** Tag for the {@link Log}. */
  private static final String TAG = "TfLiteCameraDemo";

  private static final String FRAGMENT_DIALOG = "dialog";

  private static final String HANDLE_THREAD_NAME = "CameraBackground";

  private static final int PERMISSIONS_REQUEST_CODE = 1;

  private final Object lock = new Object();
  private boolean runsegmentor = false;
  private boolean checkedPermissions = false;
  private TextView textView;
  private NumberPicker np;
  public ImageSegmentor segmentor;
  private ListView deviceView;
  private ListView filterView;
  public SeekBar seekBar;
  public GPUImageView gpuImageView;
  public GPUImageToneCurveFilter curve_filter;
  InputStream is=null;
  public Bitmap bgd, bgd1, bgd2, bgd3 = null;
  public Boolean init=false;
  public int filter_idx=0;
  public static int mskthresh=50;
  public static Net net;
  public static int tvwidth, tvheight;

  //Renderscript
  private static ScriptC_saturation saturation;
  private static android.support.v8.renderscript.RenderScript rs;


  /** Max preview width that is guaranteed by Camera2 API */
  private static final int MAX_PREVIEW_WIDTH = 1920;

  /** Max preview height that is guaranteed by Camera2 API */
  private static final int MAX_PREVIEW_HEIGHT = 1080;

  /**
   * {@link TextureView.SurfaceTextureListener} handles several lifecycle events on a {@link
   * TextureView}.
   */
  private final TextureView.SurfaceTextureListener surfaceTextureListener =
      new TextureView.SurfaceTextureListener() {

        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture texture, int width, int height) {
          openCamera(width, height);
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture texture, int width, int height) {
          configureTransform(width, height);
        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture texture) {
          return true;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture texture) {}
      };

  // Model parameter constants.
  private String gpu;
  private String cpu;
  private String nnApi;
  private String videoBokeh;
  private String portraitSeg;
  private String colorTrans;
  private String renderMerge;



  /** ID of the current {@link CameraDevice}. */
  private String cameraId;

  /** An {@link AutoFitTextureView} for camera preview. */
  public static AutoFitTextureView textureView;

  /** A {@link CameraCaptureSession } for camera preview. */
  private CameraCaptureSession captureSession;

  /** A reference to the opened {@link CameraDevice}. */
  private CameraDevice cameraDevice;

  /** The {@link android.util.Size} of camera preview. */
  private Size previewSize;

  /** {@link CameraDevice.StateCallback} is called when {@link CameraDevice} changes its state. */
  private final CameraDevice.StateCallback stateCallback =
      new CameraDevice.StateCallback() {

        @Override
        public void onOpened(@NonNull CameraDevice currentCameraDevice) {
          // This method is called when the camera is opened.  We start camera preview here.
          cameraOpenCloseLock.release();
          cameraDevice = currentCameraDevice;
          createCameraPreviewSession();
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice currentCameraDevice) {
          cameraOpenCloseLock.release();
          currentCameraDevice.close();
          cameraDevice = null;
        }

        @Override
        public void onError(@NonNull CameraDevice currentCameraDevice, int error) {
          cameraOpenCloseLock.release();
          currentCameraDevice.close();
          cameraDevice = null;
          Activity activity = getActivity();
          if (null != activity) {
            activity.finish();
          }
        }
      };

  private ArrayList<String> deviceStrings = new ArrayList<String>();
  private ArrayList<String> filterStrings = new ArrayList<String>();

  /** Current indices of device and model. */
  int currentDevice = -1;

  int currentFilter = -1;

  int currentNumThreads = -1;

  int mode = 1;

  /** An additional thread for running tasks that shouldn't block the UI. */
  private HandlerThread backgroundThread;

  /** A {@link Handler} for running tasks in the background. */
  private Handler backgroundHandler;

  /** An {@link ImageReader} that handles image capture. */
  private ImageReader imageReader;

  /** {@link CaptureRequest.Builder} for the camera preview */
  private CaptureRequest.Builder previewRequestBuilder;

  /** {@link CaptureRequest} generated by {@link #previewRequestBuilder} */
  private CaptureRequest previewRequest;

  /** A {@link Semaphore} to prevent the app from exiting before closing the camera. */
  private Semaphore cameraOpenCloseLock = new Semaphore(1);

  /** A {@link CameraCaptureSession.CaptureCallback} that handles events related to capture. */
  private CameraCaptureSession.CaptureCallback captureCallback =
      new CameraCaptureSession.CaptureCallback() {

        @Override
        public void onCaptureProgressed(
            @NonNull CameraCaptureSession session,
            @NonNull CaptureRequest request,
            @NonNull CaptureResult partialResult) {}

        @Override
        public void onCaptureCompleted(
            @NonNull CameraCaptureSession session,
            @NonNull CaptureRequest request,
            @NonNull TotalCaptureResult result) {}
      };

  /**
   * Shows a {@link Toast} on the UI thread for the segmentation results.
   *
   * @param text The message to show
   */
  private void showToast(String s) {
    SpannableStringBuilder builder = new SpannableStringBuilder();
    SpannableString str1 = new SpannableString(s);
    builder.append(str1);
    showToast(builder);
  }

  private void showToast(SpannableStringBuilder builder) {
    final Activity activity = getActivity();
    if (activity != null) {
      activity.runOnUiThread(
          new Runnable() {
            @Override
            public void run() {
              textView.setText(builder, TextView.BufferType.SPANNABLE);
            }
          });
    }
  }

  /**
   * Resizes image.
   *
   * Attempting to use too large a preview size could  exceed the camera bus' bandwidth limitation,
   * resulting in gorgeous previews but the storage of garbage capture data.
   *
   * Given {@code choices} of {@code Size}s supported by a camera, choose the smallest one that is
   * at least as large as the respective texture view size, and that is at most as large as the
   * respective max size, and whose aspect ratio matches with the specified value. If such size
   * doesn't exist, choose the largest one that is at most as large as the respective max size, and
   * whose aspect ratio matches with the specified value.
   *
   * @param choices The list of sizes that the camera supports for the intended output class
   * @param textureViewWidth The width of the texture view relative to sensor coordinate
   * @param textureViewHeight The height of the texture view relative to sensor coordinate
   * @param maxWidth The maximum width that can be chosen
   * @param maxHeight The maximum height that can be chosen
   * @param aspectRatio The aspect ratio
   * @return The optimal {@code Size}, or an arbitrary one if none were big enough
   */
  private static Size chooseOptimalSize(
      Size[] choices,
      int textureViewWidth,
      int textureViewHeight,
      int maxWidth,
      int maxHeight,
      Size aspectRatio) {

    // Collect the supported resolutions that are at least as big as the preview Surface
    List<Size> bigEnough = new ArrayList<>();
    // Collect the supported resolutions that are smaller than the preview Surface
    List<Size> notBigEnough = new ArrayList<>();
    int w = aspectRatio.getWidth();
    int h = aspectRatio.getHeight();
    for (Size option : choices) {
      if (option.getWidth() <= maxWidth
          && option.getHeight() <= maxHeight
          && option.getHeight() == option.getWidth() * h / w) {
        if (option.getWidth() >= textureViewWidth && option.getHeight() >= textureViewHeight) {
          bigEnough.add(option);
        } else {
          notBigEnough.add(option);
        }
      }
    }

    // Pick the smallest of those big enough. If there is no one big enough, pick the
    // largest of those not big enough.
    if (bigEnough.size() > 0) {
      return Collections.min(bigEnough, new CompareSizesByArea());
    } else if (notBigEnough.size() > 0) {
      return Collections.max(notBigEnough, new CompareSizesByArea());
    } else {
      Log.e(TAG, "Couldn't find any suitable preview size");
      return choices[0];
    }
  }

  public static Camera2BasicFragment newInstance() {
    return new Camera2BasicFragment();
  }

  /** Layout the preview and buttons. */
  @Override
  public View onCreateView(
      LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
    return inflater.inflate(R.layout.fragment_camera2_basic, container, false);
  }


  private void changeFilter(){

    final int filterIndex = filterView.getCheckedItemPosition();
    currentFilter =  filterIndex;
    String model = filterStrings.get(filterIndex);

    if (model.equals(videoBokeh)) {
      mode=0;
      bgd=bgd1;
    } else if (model.equals(portraitSeg)) {
      mode=1;
      bgd=bgd1;
    } else if (model.equals(colorTrans)){
      mode=2;
      bgd=bgd2;
    }
    else if (model.equals(renderMerge)){

      mode=3;
      bgd=bgd3;
    }

  }

  private void updateActiveModel() {
    // Get UI information before delegating to background
    final int filterIndex = filterView.getCheckedItemPosition();
    final int deviceIndex = deviceView.getCheckedItemPosition();
    final int numThreads = np.getValue();

    backgroundHandler.post(() -> {
      if (filterIndex == currentFilter && deviceIndex == currentDevice
              && numThreads == currentNumThreads) {
        return;
      }
      currentFilter =  filterIndex;
      currentDevice = deviceIndex;
      currentNumThreads = numThreads;

      // Disable segmentor while updating
      if (segmentor != null) {
        segmentor.close();
        segmentor = null;
      }

      // Lookup names of parameters.
      String model = filterStrings.get(filterIndex);
      String device = deviceStrings.get(deviceIndex);

      Log.i(TAG, "Changing model to " + model + " device " + device);


      Log.d("Current Model",model);
      // Try to load model.
      try {

       //
        segmentor = new ImageSegmentorFloatMobileUnet(getActivity());

        if (model.equals(videoBokeh)) {
          mode=0;
          bgd=bgd1;
        } else if (model.equals(portraitSeg)) {
          mode=1;
          bgd=bgd1;
        } else if (model.equals(colorTrans)){
          mode=2;
          bgd=bgd2;
        } else if (model.equals(renderMerge)){
          mode=3;
          bgd=bgd1;
        }
      } catch (IOException e) {
        Log.d(TAG, "Failed to load", e);
        segmentor = null;
      }

      // Customize the interpreter to the type of device we want to use.
      if (segmentor == null) {
        return;
      }
      segmentor.setNumThreads(numThreads);
      if (device.equals(cpu)) {
      } else if (device.equals(gpu)) {
          segmentor.useGpu();
      } else if (device.equals(nnApi)) {
        segmentor.useNNAPI();
      }
    });
  }

  /** Connect the buttons to their event handler. */
  @Override
  public void onViewCreated(final View view, Bundle savedInstanceState) {
    gpu = getString(R.string.gpu);
    cpu = getString(R.string.cpu);
    nnApi = getString(R.string.nnapi);
    videoBokeh = getString(R.string.videoBokeh);
    portraitSeg = getString(R.string.portraitSeg);
    colorTrans = getString(R.string.colorTrans);
    renderMerge = getString(R.string.renderMerge);

    // Get references to widgets.
    textureView = (AutoFitTextureView) view.findViewById(R.id.texture);
    textView = (TextView) view.findViewById(R.id.text);
    deviceView = (ListView) view.findViewById(R.id.device);
    filterView = (ListView) view.findViewById(R.id.model);

    //GPUImage
    gpuImageView = (GPUImageView) view.findViewById(R.id.gpuimageview);
    Bitmap splash = BitmapFactory.decodeResource(getResources(),R.drawable.tf);

    Bitmap newsplash=Bitmap.createScaledBitmap(
              splash,1024 ,1024 , false);
      gpuImageView.setImage(newsplash);
      splash.recycle();


    gpuImageView.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View view) {
        gpufilter();
      }
    });

    gpuImageView.setOnLongClickListener(new View.OnLongClickListener() {
      @Override
      public boolean onLongClick(View view) {
        if(segmentor != null) {

            new AsyncTask<Void, Void, Void>() {
                @Override
                protected Void doInBackground(Void... voids) {
                    segmentor.color_harmonize();
                    return null;
                }

                @Override
                protected void onPostExecute(Void aVoid) {
                    customToast("Image Saved");
                }
            }.execute();

        }
        Log.d("HARMONIZE","Called harmonizer ...");
        return true;
      }
    });


    // Read the Photoshop ACV files
    AssetManager as = this.getActivity().getAssets();
    //is = null;
     curve_filter = new GPUImageToneCurveFilter();

    try {
      is = as.open("green.acv");
       curve_filter.setFromCurveFileInputStream(is);
       is.close();
      Log.e("MainActivity", "Success ACV Loaded");
    } catch (IOException e) {
      Log.e("MainActivity", "Error");
    }

    // Seek bar to control mask threshold
    seekBar=(SeekBar)view.findViewById(R.id.seekBar);
    // perform seek bar change listener event used for getting the progress value
    seekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {

      public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
        mskthresh = progress;
      }

      public void onStartTrackingTouch(SeekBar seekBar) { }
      public void onStopTrackingTouch(SeekBar seekBar) {
        Toast.makeText(getContext(), "Seek bar progress is :" + mskthresh,
                Toast.LENGTH_SHORT).show();
      }
    });


    BitmapFactory.Options options = new BitmapFactory.Options();
    options.inScaled = false;

    // Load background images
    bgd1= BitmapFactory.decodeResource(getResources(),R.drawable.dock_vbig, options);
    bgd2= BitmapFactory.decodeResource(getResources(),R.drawable.sunset_vbig, options);
    bgd3= BitmapFactory.decodeResource(getResources(),R.drawable.taj_vbig, options);
    bgd=bgd1;

    // Build list of models
    filterStrings.add(videoBokeh);
    filterStrings.add(portraitSeg);
    filterStrings.add(colorTrans);
    filterStrings.add(renderMerge);

    // Build list of devices
    int defaultfilterIndex = 1;
    deviceStrings.add(cpu);
    deviceStrings.add(gpu);
    deviceStrings.add(nnApi);

    deviceView.setAdapter(
        new ArrayAdapter<String>(
            getContext(), R.layout.listview_row, R.id.listview_row_text, deviceStrings));
    deviceView.setChoiceMode(ListView.CHOICE_MODE_SINGLE);
    deviceView.setOnItemClickListener(
        new AdapterView.OnItemClickListener() {
          @Override
          public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
            updateActiveModel();
          }
        });
    deviceView.setItemChecked(0, true);


    filterView.setChoiceMode(ListView.CHOICE_MODE_SINGLE);
    ArrayAdapter<String> modelAdapter =
        new ArrayAdapter<>(
            getContext(), R.layout.listview_row, R.id.listview_row_text, filterStrings);
    filterView.setAdapter(modelAdapter);
    filterView.setItemChecked(defaultfilterIndex, true);
    filterView.setOnItemClickListener(
        new AdapterView.OnItemClickListener() {
          @Override
          public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
            //updateActiveModel();
            changeFilter();
          }
        });

    np = (NumberPicker) view.findViewById(R.id.np);
    np.setMinValue(1);
    np.setMaxValue(10);
    np.setWrapSelectorWheel(true);
    np.setOnValueChangedListener(
        new NumberPicker.OnValueChangeListener() {
          @Override
          public void onValueChange(NumberPicker picker, int oldVal, int newVal) {
            updateActiveModel();
          }
        });

    // Start initial model.

      // Load a caffe network.
      String proto = getPath("deploy_512.prototxt", getContext());
      String weights = getPath("harmonize_iter_200000.caffemodel", getContext());
      net = Dnn.readNetFromCaffe(proto, weights);
      net.setPreferableTarget(Dnn.DNN_TARGET_OPENCL_FP16);
      net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
      net.enableFusion(Boolean.TRUE);
      Log.i(TAG, "Network loaded successfully");

      tvheight=textureView.getHeight();
      tvwidth=textureView.getWidth();

  }

    // Upload file to storage and return a path.
    private static String getPath(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream = null;
        try {
            // Read data from assets.
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            // Return a path to file which may be read in common way.
            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            Log.i(TAG, "Failed to upload a file");
        }
        return "";
    }



  /** Load the model and labels. */
  @Override
  public void onActivityCreated(Bundle savedInstanceState) {
    super.onActivityCreated(savedInstanceState);
    startBackgroundThread();
  }

  @Override
  public void onResume() {
    super.onResume();
    startBackgroundThread();

    // When the screen is turned off and turned back on, the SurfaceTexture is already
    // available, and "onSurfaceTextureAvailable" will not be called. In that case, we can open
    // a camera and start preview from here (otherwise, we wait until the surface is ready in
    // the SurfaceTextureListener).
    if (textureView.isAvailable()) {
      openCamera(textureView.getWidth(), textureView.getHeight());
    } else {
      textureView.setSurfaceTextureListener(surfaceTextureListener);
    }
  }

  @Override
  public void onPause() {
    closeCamera();
    stopBackgroundThread();
    super.onPause();
  }

  @Override
  public void onDestroy() {
    if (segmentor != null) {
      segmentor.close();
    }
    super.onDestroy();
  }

  /**
   * Sets up member variables related to camera.
   *
   * @param width The width of available size for camera preview
   * @param height The height of available size for camera preview
   */
  private void setUpCameraOutputs(int width, int height) {
    Activity activity = getActivity();
    CameraManager manager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
    try {
      for (String cameraId : manager.getCameraIdList()) {
        CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

        // We don't use a front facing camera in this sample.
        Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
        if (facing != null && facing == CameraCharacteristics.LENS_FACING_BACK) {
          continue;
        }

        StreamConfigurationMap map =
            characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        if (map == null) {
          continue;
        }

        // // For still image captures, we use the largest available size.
        Size largest =
            Collections.max(
                Arrays.asList(map.getOutputSizes(ImageFormat.JPEG)), new CompareSizesByArea());
        imageReader =
            ImageReader.newInstance(
                largest.getWidth(), largest.getHeight(), ImageFormat.JPEG, /*maxImages*/ 2);

        // Find out if we need to swap dimension to get the preview size relative to sensor
        // coordinate.
        int displayRotation = activity.getWindowManager().getDefaultDisplay().getRotation();
        // noinspection ConstantConditions
        /* Orientation of the camera sensor */
        int sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
        boolean swappedDimensions = false;
        switch (displayRotation) {
          case Surface.ROTATION_0:
          case Surface.ROTATION_180:
            if (sensorOrientation == 90 || sensorOrientation == 270) {
              swappedDimensions = true;
            }
            break;
          case Surface.ROTATION_90:
          case Surface.ROTATION_270:
            if (sensorOrientation == 0 || sensorOrientation == 180) {
              swappedDimensions = true;
            }
            break;
          default:
            Log.e(TAG, "Display rotation is invalid: " + displayRotation);
        }

        Point displaySize = new Point();
        activity.getWindowManager().getDefaultDisplay().getSize(displaySize);
        int rotatedPreviewWidth = width;
        int rotatedPreviewHeight = height;
        int maxPreviewWidth = displaySize.x;
        int maxPreviewHeight = displaySize.y;

        if (swappedDimensions) {
          rotatedPreviewWidth = height;
          rotatedPreviewHeight = width;
          maxPreviewWidth = displaySize.y;
          maxPreviewHeight = displaySize.x;
        }

        if (maxPreviewWidth > MAX_PREVIEW_WIDTH) {
          maxPreviewWidth = MAX_PREVIEW_WIDTH;
        }

        if (maxPreviewHeight > MAX_PREVIEW_HEIGHT) {
          maxPreviewHeight = MAX_PREVIEW_HEIGHT;
        }

        previewSize =
            chooseOptimalSize(
                map.getOutputSizes(SurfaceTexture.class),
                rotatedPreviewWidth,
                rotatedPreviewHeight,
                maxPreviewWidth,
                maxPreviewHeight,
                largest);

        // We fit the aspect ratio of TextureView to the size of preview we picked.
        int orientation = getResources().getConfiguration().orientation;
        if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
          textureView.setAspectRatio(previewSize.getWidth(), previewSize.getHeight());
        } else {
          textureView.setAspectRatio(previewSize.getHeight(), previewSize.getWidth());
        }

        this.cameraId = cameraId;
        return;
      }
    } catch (CameraAccessException e) {
      Log.e(TAG, "Failed to access Camera", e);
    } catch (NullPointerException e) {
      // Currently an NPE is thrown when the Camera2API is used but not supported on the
      // device this code runs.
      ErrorDialog.newInstance(getString(R.string.camera_error))
          .show(getChildFragmentManager(), FRAGMENT_DIALOG);
    }
  }

  private String[] getRequiredPermissions() {
    Activity activity = getActivity();
    try {
      PackageInfo info =
          activity
              .getPackageManager()
              .getPackageInfo(activity.getPackageName(), PackageManager.GET_PERMISSIONS);
      String[] ps = info.requestedPermissions;
      if (ps != null && ps.length > 0) {
        return ps;
      } else {
        return new String[0];
      }
    } catch (Exception e) {
      return new String[0];
    }
  }

  /** Opens the camera specified by {@link Camera2BasicFragment#cameraId}. */
  private void openCamera(int width, int height) {
    if (!checkedPermissions && !allPermissionsGranted()) {
      FragmentCompat.requestPermissions(this, getRequiredPermissions(), PERMISSIONS_REQUEST_CODE);
      return;
    } else {
      checkedPermissions = true;
    }
    setUpCameraOutputs(width, height);
    configureTransform(width, height);
    Activity activity = getActivity();
    CameraManager manager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
    try {
      if (!cameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
        throw new RuntimeException("Time out waiting to lock camera opening.");
      }
      manager.openCamera(cameraId, stateCallback, backgroundHandler);
    } catch (CameraAccessException e) {
      Log.e(TAG, "Failed to open Camera", e);
    } catch (InterruptedException e) {
      throw new RuntimeException("Interrupted while trying to lock camera opening.", e);
    }
  }

  private boolean allPermissionsGranted() {
    for (String permission : getRequiredPermissions()) {
      if (ContextCompat.checkSelfPermission(getActivity(), permission)
          != PackageManager.PERMISSION_GRANTED) {
        return false;
      }
    }
    return true;
  }

  @Override
  public void onRequestPermissionsResult(
      int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
  }

  /** Closes the current {@link CameraDevice}. */
  private void closeCamera() {
    try {
      cameraOpenCloseLock.acquire();
      if (null != captureSession) {
        captureSession.close();
        captureSession = null;
      }
      if (null != cameraDevice) {
        cameraDevice.close();
        cameraDevice = null;
      }
      if (null != imageReader) {
        imageReader.close();
        imageReader = null;
      }
    } catch (InterruptedException e) {
      throw new RuntimeException("Interrupted while trying to lock camera closing.", e);
    } finally {
      cameraOpenCloseLock.release();
    }
  }

  /** Starts a background thread and its {@link Handler}. */
  private void startBackgroundThread() {
    backgroundThread = new HandlerThread(HANDLE_THREAD_NAME);
    backgroundThread.start();
    backgroundHandler = new Handler(backgroundThread.getLooper());
    // Start the segmentation train & load an initial model.
    synchronized (lock) {
      runsegmentor = true;
    }
    backgroundHandler.post(periodicSegment);
    updateActiveModel();

    //To use gpu default
    filterView.setItemChecked(1, true);
    deviceView.setItemChecked(1, true);

    // Renderscript initialization
    rs = android.support.v8.renderscript.RenderScript.create(this.getActivity());
    saturation = new ScriptC_saturation(rs);

  }

  /** Stops the background thread and its {@link Handler}. */
  private void stopBackgroundThread() {
    backgroundThread.quitSafely();
    try {
      backgroundThread.join();
      backgroundThread = null;
      backgroundHandler = null;
      synchronized (lock) {
        runsegmentor = false;
      }
    } catch (InterruptedException e) {
      Log.e(TAG, "Interrupted when stopping background thread", e);
    }
  }
  /** Takes photos and Segment them periodically. */
  private Runnable periodicSegment =
      new Runnable() {
        @Override
        public void run() {
          synchronized (lock) {
            if (runsegmentor) {
              segmentFrame();

          }
          }
          backgroundHandler.post(periodicSegment);
        }
      };

  /** Creates a new {@link CameraCaptureSession} for camera preview. */
  private void createCameraPreviewSession() {
    try {
      SurfaceTexture texture = textureView.getSurfaceTexture();
      assert texture != null;

      // We configure the size of default buffer to be the size of camera preview we want.
      texture.setDefaultBufferSize(previewSize.getWidth(), previewSize.getHeight());

      // This is the output Surface we need to start preview.
      Surface surface = new Surface(texture);

      // We set up a CaptureRequest.Builder with the output Surface.
      previewRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
      previewRequestBuilder.addTarget(surface);

      // Here, we create a CameraCaptureSession for camera preview.
      cameraDevice.createCaptureSession(
          Arrays.asList(surface),
          new CameraCaptureSession.StateCallback() {

            @Override
            public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
              // The camera is already closed
              if (null == cameraDevice) {
                return;
              }

              // When the session is ready, we start displaying the preview.
              captureSession = cameraCaptureSession;
              try {
                // Auto focus should be continuous for camera preview.
                previewRequestBuilder.set(
                    CaptureRequest.CONTROL_AF_MODE,
                    CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

                // Finally, we start displaying the camera preview.
                previewRequest = previewRequestBuilder.build();
                captureSession.setRepeatingRequest(
                    previewRequest, captureCallback, backgroundHandler);
              } catch (CameraAccessException e) {
                Log.e(TAG, "Failed to set up config to capture Camera", e);
              }
            }

            @Override
            public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {
              showToast("Failed");
            }
          },
          null);
    } catch (CameraAccessException e) {
      Log.e(TAG, "Failed to preview Camera", e);
    }
  }

  /**
   * Configures the necessary {@link android.graphics.Matrix} transformation to `textureView`. This
   * method should be called after the camera preview size is determined in setUpCameraOutputs and
   * also the size of `textureView` is fixed.
   *
   * @param viewWidth The width of `textureView`
   * @param viewHeight The height of `textureView`
   */
  private void configureTransform(int viewWidth, int viewHeight) {
    Activity activity = getActivity();
    if (null == textureView || null == previewSize || null == activity) {
      return;
    }
    int rotation = activity.getWindowManager().getDefaultDisplay().getRotation();
    Matrix matrix = new Matrix();
    RectF viewRect = new RectF(0, 0, viewWidth, viewHeight);
    RectF bufferRect = new RectF(0, 0, previewSize.getHeight(), previewSize.getWidth());
    float centerX = viewRect.centerX();
    float centerY = viewRect.centerY();
    if (Surface.ROTATION_90 == rotation || Surface.ROTATION_270 == rotation) {
      bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());
      matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL);
      float scale =
          Math.max(
              (float) viewHeight / previewSize.getHeight(),
              (float) viewWidth / previewSize.getWidth());
      matrix.postScale(scale, scale, centerX, centerY);
      matrix.postRotate(90 * (rotation - 2), centerX, centerY);
    } else if (Surface.ROTATION_180 == rotation) {
      matrix.postRotate(180, centerX, centerY);
    }
    textureView.setTransform(matrix);
  }

  /** Segments a frame from the preview stream. */
  private void segmentFrame() {
    if (segmentor == null || getActivity() == null || cameraDevice == null) {
      // It's important to not call showToast every frame, or else the app will starve and
      // hang. updateActiveModel() already puts an error message up with showToast.
      // showToast("Uninitialized segmentor or invalid context.");
      return;
    }
    SpannableStringBuilder textToShow = new SpannableStringBuilder();
    Bitmap bitmap = textureView.getBitmap(segmentor.getImageSizeX(), segmentor.getImageSizeY());
    Bitmap fgd = textureView.getBitmap();
    bgd=Bitmap.createScaledBitmap(
              bgd,textureView.getWidth() ,textureView.getHeight() , false);
    segmentor.segmentFrame(bitmap, mode, fgd, bgd);


    Log.d("TV height", String.valueOf(textureView.getHeight()));
    Log.d("TV width", String.valueOf(textureView.getWidth()));

    bitmap.recycle();
    showToast(filterStrings.get(mode)+"    Frame Rate: "+(1000/segmentor.duration));

    if(!init) {
      // Delete loading screen
      gpuImageView.getGPUImage().deleteImage();
      init= true;
    }

    new Handler(Looper.getMainLooper()).post(new Runnable() {
      @Override
      public void run() {

        if(segmentor!=null && segmentor.result!=null)
        gpuImageView.setImage(segmentor.result); // this loads image on the current thread, should be run in a thread

      }
    });


  }

  public void customToast(String message){

      Toast toast = Toast.makeText(getContext(), message, Toast.LENGTH_SHORT);

      // Set the toast position
      toast.setGravity(Gravity.TOP|Gravity.CENTER_HORIZONTAL, 0, 0);
      // Show toast
      toast.show();
  }

  public void gpufilter() {

    //Roate filters cyclically
    if (filter_idx>3)
      filter_idx=0;
    else
      filter_idx=filter_idx+1;

    switch (filter_idx) {


      case 0:{
          // Clear all filters
        gpuImageView.setFilter(new GPUImageFilter());
        break;
      }
      case 1:{
          // Apply sepia filter
       gpuImageView.setFilter(new GPUImageSepiaToneFilter());
          customToast("Sepia");
       break;
      }
      case 2:{
          // Apply emboss filter
        gpuImageView.setFilter(new GPUImageEmbossFilter());
          customToast("Emboss");
        break;
       }
      case 3:{
        // Add photoshop acv curve filters
        gpuImageView.setFilter(curve_filter);
          customToast("Greeny");
        break;
      }
      case 4:{
          // Add multiple filters
          GPUImageFilterGroup filterGroup = new GPUImageFilterGroup();
          filterGroup.addFilter(new GPUImageContrastFilter(1.5f));
          filterGroup.addFilter(new GPUImageKuwaharaFilter(4));
          gpuImageView.setFilter(filterGroup);
          customToast("Kuwahara");
        break;
      }

    }
  }


  public static Bitmap renderSmooth(Bitmap bgd, Bitmap fgd, Bitmap msk) {


    Bitmap output = Bitmap.createBitmap(bgd.getWidth(), bgd.getHeight(), bgd.getConfig());

    android.support.v8.renderscript.Allocation bgdAllocation = android.support.v8.renderscript.Allocation.createFromBitmap(rs,bgd);
    android.support.v8.renderscript.Allocation fgdAllocation = android.support.v8.renderscript.Allocation.createFromBitmap(rs,fgd);
    android.support.v8.renderscript.Allocation mskAllocation = android.support.v8.renderscript.Allocation.createFromBitmap(rs,msk);
    android.support.v8.renderscript.Allocation outputAllocation = android.support.v8.renderscript.Allocation.createFromBitmap(rs,output);

    saturation.set_fgd_alloc(fgdAllocation);
    saturation.set_mask_alloc(mskAllocation);
    saturation.forEach_saturation(bgdAllocation,outputAllocation);
    outputAllocation.copyTo(output);

    return output;
  }

  /** Compares two {@code Size}s based on their areas. */
  private static class CompareSizesByArea implements Comparator<Size> {

    @Override
    public int compare(Size lhs, Size rhs) {
      // We cast here to ensure the multiplications won't overflow
      return Long.signum(
          (long) lhs.getWidth() * lhs.getHeight() - (long) rhs.getWidth() * rhs.getHeight());
    }
  }

  /** Shows an error message dialog. */
  public static class ErrorDialog extends DialogFragment {

    private static final String ARG_MESSAGE = "message";

    public static ErrorDialog newInstance(String message) {
      ErrorDialog dialog = new ErrorDialog();
      Bundle args = new Bundle();
      args.putString(ARG_MESSAGE, message);
      dialog.setArguments(args);
      return dialog;
    }

    @Override
    public Dialog onCreateDialog(Bundle savedInstanceState) {
      final Activity activity = getActivity();
      return new AlertDialog.Builder(activity)
          .setMessage(getArguments().getString(ARG_MESSAGE))
          .setPositiveButton(
              android.R.string.ok,
              new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialogInterface, int i) {
                  activity.finish();
                }
              })
          .create();
    }
  }
}
