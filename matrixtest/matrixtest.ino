#include <BasicLinearAlgebra.h>

using namespace BLA;

void setup()
{
    Serial.begin(9600);

    BLA::Matrix<3, 3> A;

    BLA::Matrix<3> v;

    v.Fill(0);

    v(2, 0) = 5.3;
    v(1) = 43.67;

    A = {3.25, 5.67, 8.67, 4.55, 7.23, 9.00, 2.35, 5.73, 10.56};

    BLA::Matrix<3, 3> B = {6.54, 3.66, 2.95, 3.22, 7.54, 5.12, 8.98, 9.99, 1.56};

    v.Cols;
    v.Rows;

    BLA::Matrix<3, 3> C = A + B;

    C -= B;

    B += A;

    BLA::Matrix<3, 1> D = A * v;

    BLA::Matrix<1, 3> D_T = ~D;

    BLA::Matrix<3, 6> AleftOfB = A || B;

    BLA::Matrix<6, 3> AonTopOfB = A && B;

    BLA::Matrix<3, 3> C_inv = C;
    bool is_nonsingular = Invert(C_inv);

    Serial << "v(1): " << v(1) << '\n';

    Serial << "B: " << B << '\n';

    Serial << "identity matrix: " << AleftOfB * AonTopOfB - (A * A + B * B) + C * C_inv;

    BLA::Matrix<3> x;
    BLA::Matrix<2> u;
    BLA::Matrix<3, 2> G;
    BLA::Matrix<3, 3> F;
    float dt;
    x += (F * x + G * u) * dt;
}

void loop() {}