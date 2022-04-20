#ifndef _TRANSFORM_H_
#define _TRANSFORM_H_

#include <iostream>
#include <string>
#include <Eigen/Eigen>

template <typename FloatType> class Rigid3
{
public:
  
  using Vector = Eigen::Matrix<FloatType, 3, 1>;
  using Quaternion = Eigen::Quaternion<FloatType>;
  using AngleAxis = Eigen::AngleAxis<FloatType>;

  Rigid3() : translation_(Vector::Zero()), rotation_(Quaternion::Identity()) {}
  Rigid3(const Vector& translation, const Quaternion& rotation)
      : translation_(translation), rotation_(rotation)
  {
  }
  Rigid3(const Vector& translation, const AngleAxis& rotation)
      : translation_(translation), rotation_(rotation)
  {
  }
  static Rigid3 Rotation(const AngleAxis& angle_axis)
  {
    return Rigid3(Vector::Zero(), Quaternion(angle_axis));
  }

  static Rigid3 Rotation(const Quaternion& rotation)
  {
    return Rigid3(Vector::Zero(), rotation);
  }

  static Rigid3 Translation(const Vector& vector)
  {
    return Rigid3(vector, Quaternion::Identity());
  }

  static Rigid3 FromArrays(const std::array<FloatType, 4>& rotation,
                           const std::array<FloatType, 3>& translation)
  {
    return Rigid3(Eigen::Map<const Vector>(translation.data()),
                  Eigen::Quaternion<FloatType>(rotation[0], rotation[1],
                                               rotation[2], rotation[3]));
  }

  static Rigid3<FloatType> Identity() { return Rigid3<FloatType>(); }

  template <typename OtherType> Rigid3<OtherType> cast() const
  {
    return Rigid3<OtherType>(translation_.template cast<OtherType>(),
                             rotation_.template cast<OtherType>());
  }

  const Vector& translation() const { return translation_; }
  const Quaternion& rotation() const { return rotation_; }

  Rigid3 inverse() const
  {
    const Quaternion rotation = rotation_.conjugate();
    const Vector translation = -(rotation * translation_);
    return Rigid3(translation, rotation);
  }

  bool IsValid() const
  {
    return !std::isnan(translation_.x()) && !std::isnan(translation_.y()) &&
           !std::isnan(translation_.z()) &&
           std::abs(FloatType(1) - rotation_.norm()) < FloatType(1e-3);
  }
  Eigen::Transform<FloatType, 3, Eigen::Affine> ToAffine()
  {
    Eigen::Transform<FloatType, 3, Eigen::Affine> pose;
    pose = Eigen::Translation<FloatType, 3>(translation_.x(), translation_.y(),
                                            translation_.z()) *
           rotation_;
    return pose;
  };

  FloatType getYaw()
  {
    const Eigen::Matrix<FloatType, 3, 1> direction =
        rotation_ * Eigen::Matrix<FloatType, 3, 1>::UnitX();
    return atan2(direction.y(), direction.x());
  }
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  Vector translation_;
  Quaternion rotation_;
};
template <typename FloatType>
Rigid3<FloatType> operator*(const Rigid3<FloatType>& lhs,
                            const Rigid3<FloatType>& rhs)
{
  return Rigid3<FloatType>(lhs.rotation() * rhs.translation() +
                               lhs.translation(),
                           (lhs.rotation() * rhs.rotation()).normalized());
}

template <typename FloatType>
typename Rigid3<FloatType>::Vector
operator*(const Rigid3<FloatType>& rigid,
          const typename Rigid3<FloatType>::Vector& point)
{
  return rigid.rotation() * point + rigid.translation();
}

// This is needed for gmock.
template <typename T>
std::ostream& operator<<(std::ostream& os, const Rigid3<T>& rigid)
{
  os << "\033[1m\033[36m"
     << "t:[" << rigid.translation().x() << "," << rigid.translation().y()
     << "," << rigid.translation().z() << "]  quat:[" << rigid.rotation().w()
     << "," << rigid.rotation().x() << "," << rigid.rotation().y() << ","
     << rigid.rotation().z() << "]"
     << "\033[0m";
  return os;
}

using Rigid3d = Rigid3<double>;
using Rigid3f = Rigid3<float>;

// Converts (roll, pitch, yaw) to a unit length quaternion. Based on the URDF
// specification http://wiki.ros.org/urdf/XML/joint.
template <typename T>
Eigen::Quaternion<T> RollPitchYaw(const T& roll, const T& pitch, const T& yaw)
{
  const Eigen::AngleAxis<T> roll_angle(roll, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxis<T> pitch_angle(pitch, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxis<T> yaw_angle(yaw, Eigen::Vector3d::UnitZ());
  return yaw_angle * pitch_angle * roll_angle;
}
template <typename FloatType>
Eigen::Matrix<FloatType, 3, 1>
quatToEuler(const Eigen::Quaternion<FloatType>& rotation2)
{
  Eigen::Matrix<FloatType, 3, 1> euler_out(FloatType(0.0), FloatType(0.0),
                                           FloatType(0.0));
  Eigen::Matrix<FloatType, 3, 1> euler_out2(FloatType(0.0), FloatType(0.0),
                                            FloatType(0.0)); // second solution

  {
    Eigen::Quaternion<FloatType> rotation = rotation2.normalized();
    // get m_el
    Eigen::Matrix<FloatType, 3, 1> m_el[3];
    {
      FloatType s = FloatType(2.0);
      FloatType xs = rotation.x() * s, ys = rotation.y() * s,
                zs = rotation.z() * s;
      FloatType wx = rotation.w() * xs, wy = rotation.w() * ys,
                wz = rotation.w() * zs;
      FloatType xx = rotation.x() * xs, xy = rotation.x() * ys,
                xz = rotation.x() * zs;
      FloatType yy = rotation.y() * ys, yz = rotation.y() * zs,
                zz = rotation.z() * zs;

      m_el[0] = Eigen::Matrix<FloatType, 3, 1>(FloatType(1.0) - (yy + zz),
                                               xy - wz, xz + wy);
      m_el[1] = Eigen::Matrix<FloatType, 3, 1>(
          xy + wz, FloatType(1.0) - (xx + zz), yz - wx);
      m_el[2] = Eigen::Matrix<FloatType, 3, 1>(xz - wy, yz + wx,
                                               FloatType(1.0) - (xx + yy));
    }
    // Check that pitch is not at a singularity
    // Check that pitch is not at a singularity
    if (std::fabs(m_el[2].x()) >= 1)
    {
      euler_out(0) = 0;
      euler_out2(0) = 0;

      // From difference of angles formula
      if (m_el[2].x() < 0) // gimbal locked down
      {
        FloatType delta = atan2(m_el[0].y(), m_el[0].z());
        euler_out(1) = M_PI / FloatType(2.0);
        euler_out2(1) = M_PI / FloatType(2.0);
        euler_out(2) = delta;
        euler_out2(2) = delta;
      }
      else // gimbal locked up
      {
        FloatType delta = atan2(-m_el[0].y(), -m_el[0].z());
        euler_out(1) = -M_PI / FloatType(2.0);
        euler_out2(1) = -M_PI / FloatType(2.0);
        euler_out(2) = delta;
        euler_out2(2) = delta;
      }
    }
    else
    {
      FloatType temp_x = m_el[2].x();
      if (m_el[2].x() < FloatType(-1))
        temp_x = FloatType(-1);
      if (m_el[2].x() > FloatType(1))
        temp_x = FloatType(1);
      euler_out(1) = -asin(temp_x);
      euler_out2(1) = M_PI - euler_out(1);

      euler_out(2) = atan2(m_el[2].y() / cos(euler_out(1)),
                           m_el[2].z() / cos(euler_out(1)));
      euler_out2(2) = atan2(m_el[2].y() / cos(euler_out2(1)),
                            m_el[2].z() / cos(euler_out2(1)));

      euler_out(0) = atan2(m_el[1].x() / cos(euler_out(1)),
                           m_el[0].x() / cos(euler_out(1)));
      euler_out2(0) = atan2(m_el[1].x() / cos(euler_out2(1)),
                            m_el[0].x() / cos(euler_out2(1)));
    }
  }
  return euler_out;
}

// Returns the non-negative rotation angle in radians of the 3D transformation
// 'transform'.
template <typename FloatType>
FloatType GetAngle(const Rigid3<FloatType>& transform)
{
  return FloatType(2) * std::atan2(transform.rotation().vec().norm(),
                                   std::abs(transform.rotation().w()));
}

// Returns the yaw component in radians of the given 3D 'rotation'. Assuming
// 'rotation' is composed of three rotations around X, then Y, then Z, returns
// the angle of the Z rotation.
template <typename T> T GetYaw(const Eigen::Quaternion<T>& rotation)
{
  const Eigen::Matrix<T, 3, 1> direction =
      rotation * Eigen::Matrix<T, 3, 1>::UnitX();
  return atan2(direction.y(), direction.x());
}

// Returns the yaw component in radians of the given 3D transformation
// 'transform'.
template <typename T> T GetYaw(const Rigid3<T>& transform)
{
  return GetYaw(transform.rotation());
}

// Returns an angle-axis vector (a vector with the length of the rotation angle
// pointing to the direction of the rotation axis) representing the same
// rotation as the given 'quaternion'.
template <typename T>
Eigen::Matrix<T, 3, 1>
RotationQuaternionToAngleAxisVector(const Eigen::Quaternion<T>& quaternion)
{
  Eigen::Quaternion<T> normalized_quaternion = quaternion.normalized();
  // We choose the quaternion with positive 'w', i.e., the one with a smaller
  // angle that represents this orientation.
  if (normalized_quaternion.w() < 0.)
  {
    // Multiply by -1. http://eigen.tuxfamily.org/bz/show_bug.cgi?id=560
    normalized_quaternion.w() = -1. * normalized_quaternion.w();
    normalized_quaternion.x() = -1. * normalized_quaternion.x();
    normalized_quaternion.y() = -1. * normalized_quaternion.y();
    normalized_quaternion.z() = -1. * normalized_quaternion.z();
  }
  // We convert the normalized_quaternion into a vector along the rotation axis
  // with length of the rotation angle.
  const T angle =
      2. * atan2(normalized_quaternion.vec().norm(), normalized_quaternion.w());
  constexpr double kCutoffAngle = 1e-7; // We linearize below this angle.
  const T scale = angle < kCutoffAngle ? T(2.) : angle / sin(angle / 2.);
  return Eigen::Matrix<T, 3, 1>(scale * normalized_quaternion.x(),
                                scale * normalized_quaternion.y(),
                                scale * normalized_quaternion.z());
}

// Returns a quaternion representing the same rotation as the given 'angle_axis'
// vector.
template <typename T>
Eigen::Quaternion<T>
AngleAxisVectorToRotationQuaternion(const Eigen::Matrix<T, 3, 1>& angle_axis)
{
  T scale = T(0.5);
  T w = T(1.);
  constexpr double kCutoffAngle = 1e-8; // We linearize below this angle.
  if (angle_axis.squaredNorm() > kCutoffAngle)
  {
    const T norm = angle_axis.norm();
    scale = sin(norm / 2.) / norm;
    w = cos(norm / 2.);
  }
  const Eigen::Matrix<T, 3, 1> quaternion_xyz = scale * angle_axis;
  return Eigen::Quaternion<T>(w, quaternion_xyz.x(), quaternion_xyz.y(),
                              quaternion_xyz.z());
}

struct ConstraintPose
{
  Rigid3d zbar_ij;
  double translation_weight;
  double rotation_weight;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <typename T> struct TimestampedTransform
{
  double time;
  Rigid3<T> transform;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <typename T>
Rigid3<T> Interpolate(const TimestampedTransform<T>& start,
                      const TimestampedTransform<T>& end, const double& time)
{
  const double duration = (end.time - start.time);
  const double factor = (time - start.time) / duration;
  const typename Rigid3<T>::Vector origin =
      start.transform.translation() +
      (end.transform.translation() - start.transform.translation()) * factor;
  const typename Rigid3<T>::Quaternion rotation =
      Eigen::Quaterniond(start.transform.rotation())
          .slerp(factor, Eigen::Quaterniond(end.transform.rotation()));
  return Rigid3<T>(origin, rotation);
}
// // Projects 'transform' onto the XY plane.
// template <typename T>
// Rigid2<T> Project2D(const Rigid3<T>& transform) {
//   return Rigid2<T>(transform.translation().template head<2>(),
//                    GetYaw(transform));
// }
//
// // Embeds 'transform' into 3D space in the XY plane.
// template <typename T>
// Rigid3<T> Embed3D(const Rigid2<T>& transform) {
//   return Rigid3<T>(
//       {transform.translation().x(), transform.translation().y(), T(0)},
//       Eigen::AngleAxis<T>(transform.rotation().angle(),
//                           Eigen::Matrix<T, 3, 1>::UnitZ()));
// }
#endif //_TRANSFORM_H_