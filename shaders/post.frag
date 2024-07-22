#version 450

layout(binding = 0) uniform sampler2D texSampler;

layout(location = 0) in vec2 outUV;

layout(location = 0) out vec4 fragColor;

void main() {
	// float gamma = 1.0f / 2.2f;
	// fragColor = pow(texture(texSampler, outUV).rgba, vec4(gamma));
	fragColor = texture(texSampler, outUV);
}
