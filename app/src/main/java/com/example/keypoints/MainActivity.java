package com.example.keypoints;
import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.database.Cursor;
import android.graphics.Bitmap;

import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.SystemClock;
import android.provider.MediaStore;

import android.util.Log;
import android.util.TypedValue;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.core.app.ActivityCompat;
import androidx.core.content.FileProvider;


import com.example.keypoints.env.Utils;

import org.json.JSONException;

import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

public class MainActivity extends Activity {

    private static final String DOG_IMAGE = "dog_face.jpg";

    private static String TF_OD_API_MODEL_FILE = "keypoints_2.tflite";

    private ImageKeypoints segmenter;

    private static final String fileName = "output.jpg";

    private Bitmap rgbFrameBitmap = null;

    long startTime, inferenceTime;

    private static final int REQUEST_IMAGE_SELECT = 200;
    private static final int REQUEST_IMAGE_CAPTURE = 0;

    private static final int REQUEST_EXTERNAL_STORAGE = 1;

    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    private File mFile;


    private String imgPath;

    public static void verifyStoragePermissions(Activity activity) {
        // Check if we have write permission
        int permission1 = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);

        if (permission1 != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission so prompt the user
            ActivityCompat.requestPermissions(
                    activity,
                    PERMISSIONS_STORAGE,
                    REQUEST_EXTERNAL_STORAGE
            );
        }
    }

    private static final String TAG = "MainActivity";

    private Button detectButton, galleryButton, cameraButton;
    private ImageView imageView;
    private TextView resultText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        verifyStoragePermissions(this);
        mFile = getPhotoFile();

        setContentView(R.layout.activity_main);
        detectButton = findViewById(R.id.detectButton);
        imageView = findViewById(R.id.imageView);
        galleryButton = findViewById(R.id.galleryButton);
        resultText = findViewById(R.id.result);
        cameraButton = findViewById(R.id.btn_camera);

        galleryButton.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                resultText.setText("");
                Intent i = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(i, REQUEST_IMAGE_SELECT);
            }
        });

        final Intent captureImage = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

        cameraButton.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                Uri uri = FileProvider.getUriForFile(MainActivity.this,
                        "com.example.keypoints.fileprovider",
                        mFile);
                captureImage.putExtra(MediaStore.EXTRA_OUTPUT, uri);
                List<ResolveInfo> cameraActivities = getPackageManager().queryIntentActivities(captureImage,
                        PackageManager.MATCH_DEFAULT_ONLY);
                for (ResolveInfo activity : cameraActivities) {
                    grantUriPermission(activity.activityInfo.packageName,uri, Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
                }
                startActivityForResult(captureImage, REQUEST_IMAGE_CAPTURE);
            }
        });


        try {
            segmenter = new ImageKeypoints(getAssets(),TF_OD_API_MODEL_FILE);
        } catch (IOException e) {
            e.printStackTrace();
        }

        this.rgbFrameBitmap = Utils.getBitmapFromAsset(MainActivity.this, DOG_IMAGE);

        imageView.setImageBitmap(this.rgbFrameBitmap);

        detectButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Handler handler = new Handler();

                new Thread(() -> {
                    startTime = SystemClock.uptimeMillis();

                    final float[][] result = segmenter.recognizeImage(rgbFrameBitmap);
//                    final List<Keypoints> end2endResults = end2endDetector.recognizeImage(rgbFrameBitmap);
                    inferenceTime = SystemClock.uptimeMillis() - startTime;
                    handler.post(new Runnable() {
                        @Override
                        public void run() {
                            try {
                                handleResult(result);
                            } catch (JSONException e) {
                                e.printStackTrace();
                            }
                        }
                    });
                }).start();
            }
        });
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {

        if ((requestCode == REQUEST_IMAGE_CAPTURE || requestCode == REQUEST_IMAGE_SELECT) && resultCode == RESULT_OK) {

            if (requestCode == REQUEST_IMAGE_CAPTURE) {
                imgPath = mFile.getPath();
            } else {
                Uri selectedImage = data.getData();
                String[] filePathColumn = {MediaStore.Images.Media.DATA};
                Cursor cursor = MainActivity.this.getContentResolver().query(selectedImage, filePathColumn, null, null, null);
                cursor.moveToFirst();
                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                imgPath = cursor.getString(columnIndex);
                cursor.close();
            }

            rgbFrameBitmap = BitmapFactory.decodeFile(imgPath);

            ExifInterface ei = null;
            try {
                ei = new ExifInterface(imgPath);
            } catch (IOException e) {
                e.printStackTrace();
            }

            int orientation = ei.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED);

            switch(orientation) {

                case ExifInterface.ORIENTATION_ROTATE_90:

                    this.rgbFrameBitmap = rotateImage(rgbFrameBitmap, 90);
                    break;

                case ExifInterface.ORIENTATION_ROTATE_180:

                    this.rgbFrameBitmap = rotateImage(rgbFrameBitmap, 180);
                    break;

                case ExifInterface.ORIENTATION_ROTATE_270:

                    this.rgbFrameBitmap = rotateImage(rgbFrameBitmap, 270);
                    break;

            }

            imageView.setImageBitmap(rgbFrameBitmap);

        } else {
            cameraButton.setEnabled(true);
            galleryButton.setEnabled(true);
        }

        super.onActivityResult(requestCode, resultCode, data);
    }

    public static Bitmap rotateImage(Bitmap source, float angle) {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }

    private void handleResult (float [][] results) throws JSONException {
        Bitmap tempBitmap = rgbFrameBitmap.copy(rgbFrameBitmap.getConfig(),true);
        final Canvas canvas = new Canvas(tempBitmap);
        final Paint paint = new Paint();
        paint.setColor(Color.RED);
        Log.i("123123",""+tempBitmap.getWidth());
        for (float[] result : results) {
            Log.i("13232",String.valueOf(result[0])+" "+result[1]);
            canvas.drawCircle(result[0]*tempBitmap.getWidth()/224,result[1]*tempBitmap.getHeight()/224,5,paint);
        }

        imageView.setImageBitmap(tempBitmap);

        StringBuilder tv = new StringBuilder();
        tv.append("\n\nInference Time: "  + String.format("%.3fs", ImageKeypoints.inferenceTime / 1000.0f));
        resultText.setText(tv);
    }


    public File getPhotoFile() {
        File filesDir = getFilesDir();
        return new File(filesDir, fileName);
    }
}

