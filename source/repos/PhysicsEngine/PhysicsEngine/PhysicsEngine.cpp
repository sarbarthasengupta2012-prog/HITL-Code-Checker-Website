#include <iostream>
#include <cmath>
#include <SFML/Graphics.hpp>

struct Vector3 {
    float float_array[3];

    float& operator[](int index) { return float_array[index]; }
    const float& operator[](int index) const { return float_array[index]; }
};

struct Matrix {
    Vector3 rows[3];

    Vector3& operator[](int index) { return rows[index]; }
    const Vector3& operator[](int index) const { return rows[index]; }
};

Matrix multiplyMatrix3x3(const Matrix& a, const Matrix& b) {
    Matrix result = { {{0,0,0},{0,0,0},{0,0,0}} };

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

Vector3 multiplyMatrixandVector(const Matrix& m, const Vector3& v) {
    Vector3 result = { {0,0,0} };

    for (int i = 0; i < 3; i++) {
        for (int k = 0; k < 3; k++) {
            result[i] += m[i][k] * v[k];
        }
    }
    return result;
}

class obj {
public:
    Vector3 vectors[8] = {
        {{-1.0f, -1.0f, -1.0f}}, //left bottom corner at the back
        {{ 1.0f, -1.0f, -1.0f}}, //right bottom corner at the back
        {{ 1.0f,  1.0f, -1.0f}}, //right upper corner at the back
        {{-1.0f,  1.0f, -1.0f}}, //left upper corner at the back
        {{-1.0f, -1.0f,  1.0f}},  //left bottom corner at the front
        {{ 1.0f, -1.0f,  1.0f}},  //right bottom corner at the front
        {{ 1.0f,  1.0f,  1.0f}},  //right upper corner at the front
        {{-1.0f,  1.0f,  1.0f}}  //left upper corner at the front
    };

    int connections[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, // Back face
        {4, 5}, {5, 6}, {6, 7}, {7, 4}, // Front face
        {0, 4}, {1, 5}, {2, 6}, {3, 7} // Side faces
    };

    void display() const {
        for (int i = 0; i < 8; i++) {
            std::cout << "("
                << vectors[i][0] << ", "
                << vectors[i][1] << ", "
                << vectors[i][2] << ")\n";
        }
    }

    void rotateX(float theta) {
        Matrix rx = { {
            {{1, 0, 0}},
            {{0, std::cos(theta), -std::sin(theta)}},
            {{0, std::sin(theta),  std::cos(theta)}}
        } };

        for (int i = 0; i < 8; i++) {
            vectors[i] = multiplyMatrixandVector(rx, vectors[i]);
        }
    }
    void rotateY(float theta) {
        Matrix ry = { {
            {{std::cos(theta), 0, std::sin(theta)}},
            {{0,1,0}},
            {{-std::sin(theta), 0,  std::cos(theta)}}
        } };

        for (int i = 0; i < 8; i++) {
            vectors[i] = multiplyMatrixandVector(ry, vectors[i]);
        }
    }
    void rotateZ(float theta) {
        Matrix ry = { {
            {{std::cos(theta), -std::sin(theta), 0}},
            {{std::sin(theta),std::cos(theta),0}},
            {{0,0,1}}
        } };

        for (int i = 0; i < 8; i++) {
            vectors[i] = multiplyMatrixandVector(ry, vectors[i]);
        }
    }
};

int main() {
    obj Cube;

    sf::RenderWindow window(sf::VideoMode({500,500}),"My sfml project");
    window.setFramerateLimit(69);

    while (window.isOpen()) {
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
        }
        window.clear();
        Cube.rotateX(0.01f);
        Cube.rotateY(0.015f);
        Cube.rotateZ(0.001f);

        for (int i = 0; i < 12; i++) {
            Vector3 p1_3d = Cube.vectors[Cube.connections[i][0]];
            Vector3 p2_3d = Cube.vectors[Cube.connections[i][1]];

            float distance = 3.0f;
            float fov = 300.0f;

            float x1_2d = (p1_3d[0] / (p1_3d[2] + distance)) * fov + 250.0f;
            float y1_2d = (p1_3d[1] / (p1_3d[2] + distance)) * fov + 250.0f;

            float x2_2d = (p2_3d[0] / (p2_3d[2] + distance)) * fov + 250.0f;
            float y2_2d = (p2_3d[1] / (p2_3d[2] + distance)) * fov + 250.0f;

            sf::VertexArray line(sf::PrimitiveType::Lines, 2);
            line[0].position = sf::Vector2f(x1_2d, y1_2d);
            line[0].color = sf::Color::Green;
            line[1].position = sf::Vector2f(x2_2d, y2_2d);
            line[1].color = sf::Color::Green;

            window.draw(line);
        }

        window.display();
    }
    return 0;
}