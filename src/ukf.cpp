#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;
  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;
  // initial state vector
  x_ = VectorXd(5);
  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.8;
  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.8;

  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;
  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;
  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;
  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;
  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

  is_initialized_ = false;
  //time in usec
  time_us_ = .0;
  //state dimension
  n_x_ = 5;
  //augmented state dimension
  n_aug_ = 7;
  //sigma point spreading parameter
  lambda_ = max(3 - n_x_, 0);
  //predicted sigma points
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);
  //weights vector
  weights_ = VectorXd(2*n_aug_+1);
}

UKF::~UKF() {}

/**
 * @param {double} angle to be normalized between [-PI, PI]
 */
void UKF::Normalize(double *angle) {
    while (*angle > M_PI)  *angle -= 2. * M_PI;
    while (*angle < -M_PI) *angle += 2. * M_PI;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  //skip measurement if type not used
  if (((meas_package.sensor_type_ == MeasurementPackage::RADAR) && !use_radar_) ||
      ((meas_package.sensor_type_ == MeasurementPackage::LASER) && !use_laser_))
      return;
  //initializing with first measurement
  if (!is_initialized_) {
    //state vector, x
    x_ << 0.3, 0.6, 0, 0, 0;
    //state covariance matrix, P
    P_ << .01, 0, 0, 0, 0,
          0, .01, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      x_ << rho * cos(phi), rho * sin(phi), 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }
    time_us_ = meas_package.timestamp_;
    //done initializing
    is_initialized_ = true;
    return;
  }

  //elapsed time between measurements, in usec
  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;
  //prediction step
  Prediction(dt);
  //update step
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }
  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
  // print the output
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /******************************************************************************
  * Generate sigma points
  ******************************************************************************/
  //augmented mean and covariance
  VectorXd x_aug = VectorXd(7);
  MatrixXd P_aug = MatrixXd(7, 7);
  //augmentation with noise
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;
  P_aug.fill(.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;
  //square root of P
  MatrixXd L = P_aug.llt().matrixL();
  //augmented sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);
  Xsig_aug.col(0) = x_aug;
  for (int i=0; i<n_aug_; i++) {
    Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

  /******************************************************************************
  * Predict sigma points
  ******************************************************************************/
  for (int i=0; i<2*n_aug_+1; i++) {
    //extracted states
    double px = Xsig_aug(0,i);
    double py = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);
    //predicted states
    double px_p, py_p;
    if (abs(yawd) > 0.001) {
      px_p = px + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = py + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else {
      px_p = px + v * delta_t * cos(yaw);
      py_p = py + v * delta_t * sin(yaw);
    }
    double v_p    = v;
    double yaw_p  = yaw + yawd * delta_t;
    double yawd_p = yawd;
    //add noise
    px_p   += nu_a * delta_t * delta_t * cos(yaw) / 2;
    py_p   += nu_a * delta_t * delta_t * sin(yaw) / 2;
    v_p    += nu_a * delta_t;
    yaw_p  += nu_yawdd * delta_t * delta_t / 2;
    yawd_p += nu_yawdd * delta_t;
    //insert into predicted sigma points matrix
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  /******************************************************************************
  * Predict mean and covariance
  ******************************************************************************/
  //set weights
  weights_(0) = lambda_ / (lambda_+n_aug_);
  weights_.tail(2*n_aug_).fill(.5/(n_aug_+lambda_));
  //predicted state mean
  x_.fill(.0);
  for (int i=0; i<2*n_aug_+1; i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }
  //predicted state covariance
  P_.fill(.0);
  for (int i=0; i<2*n_aug_+1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    Normalize(&(x_diff(3)));
    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /******************************************************************************
  * Predict measurement
  ******************************************************************************/
  //points can be directly extracted from predicted sigma points matrix
  //mean predicted measurement
  VectorXd z_pred = VectorXd(2);
  z_pred << 0, 0;
  for (int i=0; i<2*n_aug_+1; i++) {
    z_pred += weights_(i) * Xsig_pred_.col(i).head(2);
  }
  //innovation covariance matrix
  MatrixXd S = MatrixXd(2, 2);
  S.fill(.0);
  //cross correlation matrix
  MatrixXd Tc = MatrixXd(5, 2);
  Tc.fill(.0);
  for (int i=0; i<2*n_aug_+1; i++) {
    //residual
    VectorXd z_diff = Xsig_pred_.col(i).head(2) - z_pred;
    S += weights_(i) * z_diff * z_diff.transpose();
    //state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    Normalize(&(x_diff(3)));
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }
  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(2, 2);
  R << std_laspx_ * std_laspx_, 0,
       0, std_laspy_ * std_laspy_;
  S += R;
  //Kalman gain
  MatrixXd Si = S.inverse();
  MatrixXd K = Tc * Si;

  /******************************************************************************
  * Update state
  ******************************************************************************/
  VectorXd z = meas_package.raw_measurements_;
  //residual
  VectorXd z_diff = z - z_pred;
  //update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();

  /******************************************************************************
  * Calculate NIS
  ******************************************************************************/
  double NIS_las = z_diff.transpose() * Si * z_diff;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /******************************************************************************
  * Predict measurement
  ******************************************************************************/
  MatrixXd Zsig = MatrixXd(3, 2*n_aug_+1);
  //transform sigma points into measurement space
  for (int i=0; i<2*n_aug_+1; i++) {
    double px  = Xsig_pred_(0,i);
    double py  = Xsig_pred_(1,i);
    double v   = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    double vx  = cos(yaw) * v;
    double vy  = sin(yaw) * v;
    Zsig(0,i) = sqrt(px*px + py*py);
    Zsig(1,i) = atan2(py, px);
    Zsig(2,i) = (px*vx + py*vy) / Zsig(0,i);
  }
  //mean predicted measurement
  VectorXd z_pred = VectorXd(3);
  z_pred.fill(.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred += weights_(i) * Zsig.col(i);
  }
  //innovation covariance matrix
  MatrixXd S = MatrixXd(3, 3);
  S.fill(.0);
  //cross correlation matrix
  MatrixXd Tc = MatrixXd(5, 3);
  Tc.fill(.0);
  for (int i=0; i<2*n_aug_+1; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    Normalize(&(z_diff(1)));
    S += weights_(i) * z_diff * z_diff.transpose();
    //state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    Normalize(&(x_diff(3)));
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }
  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(3, 3);
  R << std_radr_ * std_radr_,   0,   0,
       0, std_radphi_ * std_radphi_, 0,
       0,  0,  std_radrd_ * std_radrd_;
  S += R;
  //Kalman gain
  MatrixXd Si = S.inverse();
  MatrixXd K = Tc * Si;

  /******************************************************************************
  * Update state
  ******************************************************************************/
  VectorXd z = meas_package.raw_measurements_;
  //residual
  VectorXd z_diff = z - z_pred;
  Normalize(&(z_diff(1)));
  //update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();

  /******************************************************************************
  * Calculate NIS
  ******************************************************************************/
  double NIS_rad = z_diff.transpose() * Si * z_diff;

}
