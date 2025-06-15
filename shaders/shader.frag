#version 450

layout(location = 0) in vec3 fragColor;
layout(binding = 1) uniform samplerCube texSampler;

layout(location = 0) out vec4 outColor;

void main() {
    // Blending algorithm exactly following the pseudocode
    float alpha = 0.25;
    float one_minus_alpha = 0.75;

    // Initialize color to zero
    vec3 color = vec3(0.0);

    // Define the 6 cubemap face directions
    vec3 cubeFaceDirections[6] = vec3[6](
        vec3(1.0, 0.0, 0.0),   // Positive X (face 0 - red)
        vec3(-1.0, 0.0, 0.0),  // Negative X (face 1 - green)
        vec3(0.0, 1.0, 0.0),   // Positive Y (face 2 - blue)
        vec3(0.0, -1.0, 0.0),  // Negative Y (face 3 - yellow)
        vec3(0.0, 0.0, 1.0),   // Positive Z (face 4 - cyan)
        vec3(0.0, 0.0, -1.0)   // Negative Z (face 5 - white)
    );

    // Process all 6 faces as in the pseudocode
    for (int i = 0; i < 6; i++) {
        // image_load v_texel, v_pixel_x, s_cubemap_resource
        vec4 texel = texture(texSampler, cubeFaceDirections[i]);

        // blend texel to color (for RGB components k = 0,1,2)
        // v_mul_f32 v_color[k], v_tmp, v_color[k]
        // v_fma_f32 v_color[k], v_tmp[1], v_texel[k], v_color[k]
        // This translates to: color[k] = alpha * color[k] + one_minus_alpha * texel[k]
        color.r = alpha * color.r + one_minus_alpha * texel.r;
        color.g = alpha * color.g + one_minus_alpha * texel.g;
        color.b = alpha * color.b + one_minus_alpha * texel.b;
    }

    // v_mov_b32 v_color[3], 1.0 (set alpha to 1.0)
    outColor = vec4(color, 1.0);
}
