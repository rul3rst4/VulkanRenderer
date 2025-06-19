#version 450

layout(location = 0) in vec2 fragTexCoord;
layout(binding = 0) uniform samplerCube cubemapSampler;

layout(location = 0) out vec4 outColor;

void main() {
    // Test if we can sample just one face first
    vec4 testColor = texture(cubemapSampler, vec3(1.0, 0.0, 0.0)); // Should be red
    
    // If sampling works, testColor should be red (1,0,0,1)
    // If not working, it will be black (0,0,0,1)
    
    // Let's also try a simple average of all faces
    vec3 cubeFaceDirections[6] = vec3[6](
        vec3(1.0, 0.0, 0.0),   // Positive X (face 0 - red)
        vec3(-1.0, 0.0, 0.0),  // Negative X (face 1 - green)
        vec3(0.0, 1.0, 0.0),   // Positive Y (face 2 - blue)
        vec3(0.0, -1.0, 0.0),  // Negative Y (face 3 - yellow)
        vec3(0.0, 0.0, 1.0),   // Positive Z (face 4 - cyan)
        vec3(0.0, 0.0, -1.0)   // Negative Z (face 5 - white)
    );

    vec3 blendedColor = vec3(0.0);
    
    // Sample all faces
    for (int i = 0; i < 6; i++) {
        vec4 faceColor = texture(cubemapSampler, cubeFaceDirections[i]);
        blendedColor += faceColor.rgb;
    }
    
    // Average the colors
    blendedColor /= 6.0;
    
    // If all goes well, this should produce a grayish color
    // Since (red + green + blue + yellow + cyan + white) / 6 = grayish
    
    outColor = vec4(blendedColor, 1.0);
} 