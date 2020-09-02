let webcam;  // Webcam iterator
let model;  //  Tensorflowjs model
let bg;    // Background image
var bgim;

// Initialize variables and parameters 
let canvas = document.getElementById('mycanvas');
let isPredicting = false;

// Loads the tfjs model
async function loadModel() {
 const model = await tf.loadLayersModel("http://localhost:5000/model.json"); //,{strict: false}
 return model;
}


// Perform mask feathering (Gaussian-blurring + Egde-smoothing)
function refine(mask) {

const refine_out = tf.tidy(() => {
 // Reshape input
 const newmask = mask.reshape([1, 128, 128, 1]);

 //Gaussian kernel of size (7,7)
 const kernel = tf.tensor4d([0.00092991, 0.00223073, 0.00416755, 0.00606375, 0.00687113, 0.00606375,
  0.00416755, 0.00223073, 0.00535124, 0.00999743, 0.01454618, 0.01648298,
  0.01454618, 0.00999743, 0.00416755, 0.00999743, 0.01867766, 0.02717584,
  0.03079426, 0.02717584, 0.01867766, 0.00606375, 0.01454618, 0.02717584,
  0.03954061, 0.04480539, 0.03954061, 0.02717584, 0.00687113, 0.01648298,
  0.03079426, 0.04480539, 0.05077116, 0.04480539, 0.03079426, 0.00606375,
  0.01454618, 0.02717584, 0.03954061, 0.04480539, 0.03954061, 0.02717584,
  0.00416755, 0.00999743, 0.01867766, 0.02717584, 0.03079426, 0.02717584,
  0.01867766
 ], [7, 7, 1, 1]);

 // Convolve the mask with kernel   
 const blurred = tf.conv2d(newmask, kernel, strides = [1, 1], padding = 'same');
 //Reshape the output
 const fb = blurred.squeeze(0) 
 //Normalize the mask  to 0..1 range
 const norm_msk =   fb.sub(fb.min()).div(fb.max().sub(fb.min()))

 // Return the result
 return smoothstep(norm_msk);

});

return refine_out;
}


/* Smooth the mask edges */
function smoothstep(x) {

const smooth_out = tf.tidy(() => {

 // Define the left and right edges 
 const edge0 = tf.scalar(0.3);
 const edge1 = tf.scalar(0.5);

 // Scale, bias and saturate x to 0..1 range
 const z = tf.clipByValue(x.sub(edge0).div(edge1.sub(edge0)), 0.0, 1.0);
 
 //Evaluate polynomial  z * z * (3 - 2 * x)
 return tf.square(z).mul(tf.scalar(3).sub(z.mul(tf.scalar(2))));

});

 
 return smooth_out ;
}

/*
 * Performs alpha blending with background, using mask.
 * Images are resized to 300x300 size.
 */
function process(image, mask) {

const blend_out = tf.tidy(() => {

 const img = image.resizeBilinear([300, 300]);
 const msk = refine(mask).resizeNearestNeighbor([300, 300]);;
 const img_crop = img.mul(msk);
 const bgd_crop = bgim.mul(tf.scalar(1.0).sub(msk));
 const result = tf.add(img_crop, bgd_crop);

 return result;
});

return blend_out;

}


/*
 * Predict output for every frame, asynchronously
 */
async function predict() {
 while (isPredicting) {

  // Capture the frame from the webcam.
  const img =  await getImage();
  const resize = img.resizeBilinear([128, 128]);
  const expdim = resize.expandDims(0);
 

  // Predict the model output
  const out = await  model.predict(expdim);

  // Threshold the output to obtain mask
  const thresh = tf.scalar(0.5);
  const msk = out.greater(thresh);
  const cst = tf.cast(msk,'float32');

  // Post-process the output and blend images
  const blend = process(img, cst);
  
  // Draw output on the canvas
  await tf.browser.toPixels(blend, canvas);
 
  // Dispose all tensors  
  blend.dispose();
  resize.dispose();
  msk.dispose();
  expdim.dispose();
  cst.dispose();
  thresh.dispose();
  out.dispose();
  img.dispose();

  // Wait for next frame
  await tf.nextFrame();
 
 }

}

/* Set up the on-start listener */
document.getElementById('start').addEventListener('click', async () => {
 isPredicting = true;
 predict();
});

/* Set up the on-stop listener */
document.getElementById('stop').addEventListener('click', () => {
 isPredicting = false;
});


/*
 *Captures a frame from the webcam and normalize them to float: 0-1.
 */
  async function getImage() {
 const img = await webcam.capture();
 const processedImg =
  tf.tidy(() => img.div(tf.scalar(255.0)));
 img.dispose();
 return processedImg;
}

/*
 * Convert background image to tensor and resize it
 */
function loadBackground() {

 bg = document.getElementById('bg1');

 const bim = tf.browser.fromPixels(bg);
 img = tf.image.resizeBilinear(bim, [300, 300]).div(tf.scalar(255.0));
 bim.dispose();
 return img;

}


/* Initialize the model and load background images */
async function init() {

 try {
  webcam = await tf.data.webcam(document.getElementById('webcam'));
 } catch (e) {
  console.log(e);
  alert("No webcam found");
 }

 model = await loadModel();

 bgim = loadBackground();

 const screenShot = await webcam.capture();
 const pred = model.predict(tf.zeros([1, 128, 128, 3]).toFloat());

 var readable_output = pred.dataSync();
 console.log(readable_output);
 console.log(model.summary());

 pred.dispose();
 screenShot.dispose();

}


/* Initialize the application */
init();


/* Draw the default image on output canvas */
var ctx = canvas.getContext('2d');
var defImg = new Image();
defImg.crossOrigin = "anonymous";
defImg.src = "http://localhost:5000/bg.jpg";

defImg.onload = function() {
 canvas.style.width = defImg.width;
 canvas.style.height = defImg.height;
 ctx.drawImage(defImg, 0, 0,defImg.width,defImg.height,0,0,300,300);
};



/* Load background image from local directory */
async function loadFile(event) {
 var image = document.getElementById('bg1');
 image.setAttribute('crossOrigin', 'anonymous');
 image.src = URL.createObjectURL(event.target.files[0]);
 image.onload = function() {
  bgim = tf.browser.fromPixels(image).resizeBilinear([300, 300]).div(tf.scalar(255.0));
 }
};


/* Download the blended image from output canvas */
function download() {
 var download = document.getElementById("download");
 var image = document.getElementById("mycanvas").toDataURL("image/png")
  .replace("image/png", "image/octet-stream");
 download.setAttribute("href", image);
}
