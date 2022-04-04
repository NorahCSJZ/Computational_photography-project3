Fisrt my pictures are these:

![1](https://user-images.githubusercontent.com/34802668/161455932-55e18235-eb71-40c6-a86a-a9f1e0b09f57.jpg)

![2](https://user-images.githubusercontent.com/34802668/161455935-48fbadcf-4459-4097-a71c-e781a7bf1aa7.jpg)

![3](https://user-images.githubusercontent.com/34802668/161455943-d81cf962-66d9-4179-b4d0-4f322bc7790d.jpg)

![4](https://user-images.githubusercontent.com/34802668/161455947-e4d0dee7-36af-4fec-a9e2-9a3edd26bde6.jpg)

1.Images
  The reasons I chose these pictures are:
  First, these pictures were shot when I did my first internship in China, I rememebered that it was the first day that I went to this office.So it is quite important for me. And second is that, these images were quite clear, there are no moving objects or some lights or some shadows, so it is easy for me to do the project. The third is that this building is quite huge, if I can combine them into one, that would be very cool.
  
2.Recover homographies
My challenge is first, how to extract the key features and how to get the matching points, I do not want to use hand-checking, so I kept look for ways to solove itI used SIFT to recover homographies.This could be used in OpenCV packages.First I used SIFT to get keypoints and features, then I used Brute-Force Matcher in opencv.Brute-force matching is very simple, starting with a key point in the first image and then performing a distance test (descriptor) to each key point in the second image, and finally returning the closest key point.Once we got the matching points, then the second challenge came, which is how to compute Affine matrix and then get homographies.Luckily, opencv has a function which could help me Affine matrix so that I could find homographies between matching points.

3.Warp the images
The first step is how to rectify image.I try to shift matrix for adapting the view, so I must set different shift parameters for each dimension, and then I used the warpPerspective function in Opencv to rectify the image.At first, I was not quite sure whether I got the meaning of the questions, so I made some experiments on a Cube image, just like that:

![S1 rectangle](https://user-images.githubusercontent.com/34802668/161470678-8749799b-70bf-44d3-83a5-5943bd600e56.png)

then I rectify that, but the picture is out of view range, just like that:

![S2 warped](https://user-images.githubusercontent.com/34802668/161470846-b0d5d76c-16b1-4094-8ec5-1b42df867139.png)

so I shifted it:

![S3 shifted](https://user-images.githubusercontent.com/34802668/161470862-3fb95f2b-d654-4e5d-a8ff-798d7f528bd9.png)


and it has told me that it is not just doing the rectication, but also shift it and change the position, in order to do the image mosiac.So here are the results for warping two images:

![Warped_1](https://user-images.githubusercontent.com/34802668/161471044-5127cb61-6aa6-4a12-8694-938f82fa4acc.png)

![Warped_2](https://user-images.githubusercontent.com/34802668/161471054-5e091446-b4e9-4cc5-85f6-28eacd93b411.png)

![Warped_3](https://user-images.githubusercontent.com/34802668/161471061-0d0afe9a-862a-4b1c-84d0-6e958c1ca205.png)

![Warped_4](https://user-images.githubusercontent.com/34802668/161471071-fed01db1-b74b-497f-b78d-4af54bb34508.png)

4.Blend images into a mosaic
i am not use one image as much as possible and fill in with the rest, I build a zero matrix which the shape is the like the target image(hyperparameter), then I compute fusion weights, like if three images overlap in a area, the weights of in each image should be 1/3.Then I put the rectified image into the mask I rectified before, multiplied the weight of every pixel, so I got the mosaic image.It has a smooth gradient, it is because I just put every rectified image into one image with different weights, but I didn't change the continuity of the pixels inside the image, so neither did the smooth gradient of the image itself.
And it just show like this:

![Mosaic_2](https://user-images.githubusercontent.com/34802668/161474145-65c275a3-dd5c-4245-a0c0-86f3ce4bd87b.png)

More interesting, I have drawn the tranformed bbox on the Mosaic image, just like that:

![2](https://user-images.githubusercontent.com/34802668/161474253-3893db34-73a0-469d-b52c-c0304103f922.png)

For the data you give to us, the result would be like these:

![Mosaic](https://user-images.githubusercontent.com/34802668/161474299-cb7edd54-7c09-4339-8890-85b1d127d01f.png)

![1](https://user-images.githubusercontent.com/34802668/161474326-a25ffac5-c478-4b0f-a596-2e0de64ffa59.png)

5.Bells and whistles:


