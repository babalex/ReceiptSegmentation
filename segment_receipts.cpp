#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

Mat GenerateConsecutiveNumbers(int nLength)
{
    Mat numbers(nLength, 1, CV_32F);
    for (int i = 0; i < nLength; i++)
    {
        numbers.at<float>(i) = i;
    }

    return numbers.clone();
}

void Fit1DGauss(Mat& data, float& mean, float& var)
{
    Mat indices = GenerateConsecutiveNumbers(data.rows);

    mean = sum(data.mul(indices))[0] / sum(data)[0];
    var = sum((indices - mean).mul(indices - mean).mul(data))[0] / sum(data)[0];
}

void Fit1DMoG(Mat& data, int nNumClusters, vector<float>& weights, vector<float>& means, vector<float>& vars, Mat& labels)
{
    Mat indices = GenerateConsecutiveNumbers(data.rows);

    // Initialization
    weights.resize(nNumClusters);
    means.resize(nNumClusters);
    vars.resize(nNumClusters);
    float fClustWidth = (float)data.rows / nNumClusters;
    for (int i = 0; i < nNumClusters; i++)
    {
        weights[i] = 1.0f / nNumClusters;
        means[i] = (0.5f + i) * fClustWidth;
        vars[i] = fClustWidth*fClustWidth;
    }

    Mat probs(data.rows, nNumClusters, CV_32F);
    vector<float> means_old;

    // E-step
    float rel_error = 1.0f;
    int nIter = 0;
    while (rel_error > 1.0e-3f && nIter < 100)
    {
        for (int i = 0; i < nNumClusters; i++)
        {
            exp(-(indices - means[i]).mul(indices - means[i]) / vars[i], probs.col(i));
            probs.col(i) *= weights[i] / sqrtf(2 * vars[i] * CV_PI);
        }
        Mat sum_prob;
        reduce(probs, sum_prob, 1, CV_REDUCE_SUM);
        sum_prob += 1.0e-10f;
        probs /= repeat(sum_prob, 1, nNumClusters);

        // M-step
        means_old = means;
        rel_error = 0.0f;
        for (int i = 0; i < nNumClusters; i++)
        {
            weights[i] = sum(data.mul(probs.col(i)))[0] / sum(data)[0];
            means[i] = sum(data.mul(indices).mul(probs.col(i)))[0] / sum(data.mul(probs.col(i)))[0];
            vars[i] = sum((indices - means[i]).mul(indices - means[i]).mul(data).mul(probs.col(i)))[0] / sum(data.mul(probs.col(i)))[0];

            rel_error = MAX(fabsf(means[i] - means_old[i]) / means_old[i], rel_error);
        }

        nIter++;
    }

    labels = Mat::zeros(data.rows, 1, CV_32S);
    for (int i = 0; i < data.rows; i++)
    {
        float fBestProb = 0.0f;
        for (int j = 0; j < nNumClusters; j++)
        {
            if (probs.at<float>(i, j) > fBestProb)
            {
                fBestProb = probs.at<float>(i, j);
                labels.at<int>(i) = j;
            }
        }
    }
}

void BuildVarDist(Mat& data, Mat& var_dist, Mat& var_left, Mat& var_right)
{
    var_left = Mat(data.rows - 1, 1, CV_32F);
	var_right = Mat(data.rows - 1, 1, CV_32F);
	Mat indices = GenerateConsecutiveNumbers(data.rows);

    float fVarLeft, fVarRight, fMeanLeft, fMeanRight, fWeightsLeft, fWeightsRight;
    for (int i = 0; i < data.rows - 1; i++)
    {
        fWeightsLeft = sum(data.rowRange(0, i + 1))[0];
        fWeightsRight = sum(data.rowRange(i + 1, data.rows))[0];
        fMeanLeft = sum(indices.rowRange(0, i + 1).mul(data.rowRange(0, i + 1)))[0] / (fWeightsLeft + 1.0e-10f);
		fMeanRight = sum(indices.rowRange(i + 1, data.rows).mul(data.rowRange(i + 1, data.rows)))[0] / (fWeightsRight + 1.0e-10f);
		fVarLeft = sum((indices.rowRange(0, i + 1) - fMeanLeft).mul(indices.rowRange(0, i + 1) - fMeanLeft).mul(data.rowRange(0, i + 1)))[0] / (fWeightsLeft + 1.0e-10f);
		fVarRight = sum((indices.rowRange(i + 1, data.rows) - fMeanRight).mul(indices.rowRange(i + 1, data.rows) - fMeanRight).mul(data.rowRange(i + 1, data.rows)))[0] / (fWeightsRight + 1.0e-10f);

		var_left.at<float>(i) = fVarLeft;
		var_right.at<float>(i) = fVarRight;
    }
	var_dist = var_left + var_right;
}

void DrawReceiptsSegmentation(Mat& src_image, vector<Rect>& receipts_bboxes, Mat& seg_image)
{
	// Draw receipts segmentation
	seg_image = src_image.clone();
	for (int i = 0; i < receipts_bboxes.size(); i++)
	{
		rectangle(seg_image, receipts_bboxes[i], CV_RGB(255, 0, 0), 3);
	}

	namedWindow("Receipts - Segmentation", WINDOW_NORMAL);
	imshow("Receipts - Segmentation", seg_image);
}

void FindCutsWithMoG(Mat& src_image, Mat& bw_image, int nNumReceipts, vector<Rect>& receipts_bboxes)
{
	int nPageWidth = bw_image.cols;
	int nPageHeight = bw_image.rows;
	int nMinReceiptWidth = 0.2f * MIN(nPageWidth, nPageHeight);

	// Generating horizontal and vertical profiles
	Mat hor_profile, vert_profile;
	reduce(~bw_image, hor_profile, 0, CV_REDUCE_SUM, CV_32F);
	reduce(~bw_image, vert_profile, 1, CV_REDUCE_SUM, CV_32F);
	int nFiltSize = 0.2f * nMinReceiptWidth;
	blur(hor_profile, hor_profile, Size(nFiltSize, 1));
	blur(vert_profile, vert_profile, Size(1, nFiltSize));

	vector<float> weights_hor;
	vector<float> means_hor;
	vector<float> vars_hor;
	Mat labels_hor;
	Fit1DMoG(Mat(hor_profile.t()), nNumReceipts, weights_hor, means_hor, vars_hor, labels_hor);

	vector<float> weights_vert;
	vector<float> means_vert;
	vector<float> vars_vert;
	Mat labels_vert;
	Fit1DMoG(vert_profile, nNumReceipts, weights_vert, means_vert, vars_vert, labels_vert);

	// Choose the best orientation for receipt split
	float fSumVarHor, fSumVarVert;
	fSumVarHor = sum(vars_hor)[0];
	fSumVarVert = sum(vars_vert)[0];

	// Find the optimal cuts and generate bounding boxes
	vector<Point> idx;
	int nStartPix = 0;
	if (fSumVarHor < fSumVarVert)
	{
		findNonZero(labels_hor.rowRange(0, labels_hor.rows - 1) != labels_hor.rowRange(1, labels_hor.rows), idx);
		idx.push_back(Point(0, src_image.cols - 1));
		for (int i = 0; i < idx.size(); i++)
		{
			receipts_bboxes.push_back(Rect(nStartPix, 0, idx[i].y - nStartPix, src_image.rows));
			nStartPix = idx[i].y + 1;
		}
	}
	else
	{
		findNonZero(labels_vert.rowRange(0, labels_vert.rows - 1) != labels_vert.rowRange(1, labels_vert.rows), idx);
		idx.push_back(Point(0, src_image.rows - 1));
		for (int i = 0; i < idx.size(); i++)
		{
			receipts_bboxes.push_back(Rect(0, nStartPix, src_image.cols, idx[i].y - nStartPix));
			nStartPix = idx[i].y + 1;
		}
	}

	// Draw receipts segmentation
	Mat seg_image;
	DrawReceiptsSegmentation(src_image, receipts_bboxes, seg_image);

	waitKey(-1);

}

float FindCutsRecursiveDivision(Mat& bw_image, int nNumReceipts, Rect& roi = Rect(), vector<Rect>& receipts_bboxes = vector<Rect>())
{

	int nPageWidth = bw_image.cols;
	int nPageHeight = bw_image.rows;
	int nMinReceiptWidth = 0.2f * MIN(nPageWidth, nPageHeight);

	if (roi.area() == 0)
	{
		roi = Rect(0, 0, nPageWidth, nPageHeight);
	}

	Mat bw_image_roi = Mat(bw_image, roi);
	// Generating horizontal and vertical profiles
	Mat hor_profile, vert_profile;
	reduce(~bw_image_roi, hor_profile, 0, CV_REDUCE_SUM, CV_32F);
	reduce(~bw_image_roi, vert_profile, 1, CV_REDUCE_SUM, CV_32F);
	int nFiltSize = 0.2f * nMinReceiptWidth;
	blur(hor_profile, hor_profile, Size(nFiltSize, 1));
	blur(vert_profile, vert_profile, Size(1, nFiltSize));

	Mat hor_var_dist, hor_var_left, hor_var_right;
	BuildVarDist(Mat(hor_profile.t()), hor_var_dist, hor_var_left, hor_var_right);
	Point minLocHor;
    double minVal;
    minMaxLoc(hor_var_dist, &minVal, NULL, &minLocHor);

	Mat hor_var_dist_eroded;
	erode(hor_var_dist, hor_var_dist_eroded, Mat::ones(7, 1, CV_32F));
    Mat minima_dist;
    distanceTransform(~(hor_var_dist == hor_var_dist_eroded & hor_var_dist < 1.2f*minVal), minima_dist, CV_DIST_L1, 3);
    Mat minima_dist_eroded;
    erode(minima_dist, minima_dist_eroded, Mat::ones(3, 1, CV_32F));
    vector<Point> minimasHor;
    findNonZero(minima_dist == minima_dist_eroded, minimasHor);
    vector<bool> keepPts(minimasHor.size(), true);
    vector<int> minimasHorFilt;
    minimasHorFilt.push_back(minimasHor[0].y);

    for (int i = 1; i < minimasHor.size(); i++)
    {
        if (minimasHor[i].y - minimasHor[i - 1].y < 10 && keepPts[i-1])
        {
            keepPts[i] = false;
        }
        else
        {
            minimasHorFilt.push_back(minimasHor[i].y);
        }
    }

	Mat vert_var_dist, vert_var_left, vert_var_right;
	BuildVarDist(vert_profile, vert_var_dist, vert_var_left, vert_var_right);
	Point minLocVert;
	minMaxLoc(vert_var_dist, NULL, NULL, &minLocVert);

	//Mat vert_var_dist_eroded;
	//erode(vert_var_dist, vert_var_dist_eroded, Mat::ones(nFiltSize, 1, CV_32F));
	//vector<Point> minimasVert;
	//findNonZero(vert_var_dist == vert_var_dist_eroded, minimasVert);

	if (nNumReceipts == 1)
	{
		receipts_bboxes.push_back(roi);
		return sqrtf(vert_var_dist.at<float>(0)) * sqrtf(hor_var_dist.at<float>(0));
	}

    Rect roi1, roi2;
    vector<vector<Rect>> receipts_bboxes_1(2 * (nNumReceipts - 1)), receipts_bboxes_2(2 * (nNumReceipts - 1));
    vector<float> cost(2 * (nNumReceipts - 1));

    for (int r = 0; r < minimasHorFilt.size(); r++)
    {
        roi1 = Rect(roi.x, roi.y, minimasHorFilt[r], roi.height);
        roi2 = Rect(roi.x + minimasHorFilt[r], roi.y, roi.width - minimasHorFilt[r], roi.height);

        float mean1, var1, mean2, var2;
        Fit1DGauss(Mat(hor_profile.colRange(0, minimasHorFilt[r]).t()), mean1, var1);
        Fit1DGauss(Mat(hor_profile.colRange(minimasHorFilt[r], roi.width).t()), mean2, var2);
        mean2 += minimasHorFilt[r];
        float fCost = exp(-MIN((mean1 - mean2)*(mean1 - mean2) / var1, (mean1 - mean2)*(mean1 - mean2) / var2));
        //float fCostHor = hor_var_dist.at<float>(minLocHor.y)*vert_var_dist.at<float>(0);
        //float fCostVert = vert_var_dist.at<float>(minLocVert.y)*hor_var_dist.at<float>(0);

        //if (fCostVert < fCostHor)
        //{
        //	roi1 = Rect(roi.x, roi.y, roi.width, minLocVert.y);
        //	roi2 = Rect(roi.x, roi.y + minLocVert.y, roi.width, roi.height - minLocVert.y);
        //}


        for (int i = 0; i < nNumReceipts - 1; i++)
        {
            cost[i] = FindCutsRecursiveDivision(bw_image, i + 1, roi1, receipts_bboxes_1[i])
                + FindCutsRecursiveDivision(bw_image, nNumReceipts - 1 - i, roi2, receipts_bboxes_2[i]);
        }
    }

	roi1 = Rect(roi.x, roi.y, roi.width, minLocVert.y);
	roi2 = Rect(roi.x, roi.y + minLocVert.y, roi.width, roi.height - minLocVert.y);
	for (int i = 0; i < nNumReceipts - 1; i++)
	{
		cost[i + nNumReceipts - 1] = FindCutsRecursiveDivision(bw_image, i + 1, roi1, receipts_bboxes_1[i + nNumReceipts - 1])
			+ FindCutsRecursiveDivision(bw_image, nNumReceipts - 1 - i, roi2, receipts_bboxes_2[i + nNumReceipts - 1]);
	}

	Point minCostLoc;
	minMaxLoc(cost, NULL, NULL, &minCostLoc);

	receipts_bboxes = receipts_bboxes_1[minCostLoc.x];
	receipts_bboxes.insert(receipts_bboxes.end(), receipts_bboxes_2[minCostLoc.x].begin(), receipts_bboxes_2[minCostLoc.x].end());

	return cost[minCostLoc.y];
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        cout << "Usage: segment_receipts <image filename> <number of receipts on page>" << endl;
		return 0;
    }

    // Read and display the source image
    string filename(argv[1]);
    int nNumReceipts = atoi(argv[2]);

    Mat src_image = imread(filename);
    namedWindow("Receipts - source image", WINDOW_NORMAL);
    imshow("Receipts - source image", src_image);


    // Adaptive thresholding of the receipt image
    Mat bw_image, gray_image;
    cvtColor(src_image, gray_image, CV_BGR2GRAY);
    adaptiveThreshold(gray_image, bw_image, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, 20);
    namedWindow("Receipts - BW image", WINDOW_NORMAL);
    imshow("Receipts - BW image", bw_image);

	vector<Rect> receipts_bboxes;
	FindCutsRecursiveDivision(bw_image, nNumReceipts, Rect(), receipts_bboxes);

	Mat seg_image;
	DrawReceiptsSegmentation(src_image, receipts_bboxes, seg_image);

	waitKey(-1);

	string output_file(argv[1]);
	output_file = output_file.substr(0, output_file.find_last_of('.')) + "_output.jpg";

	imwrite(output_file, seg_image);

    return 0;
}