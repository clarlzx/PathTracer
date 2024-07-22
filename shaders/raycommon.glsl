struct HitPayload {
	vec3 hitValue;
	vec3 rayOrigin;
	vec3 rayDirection;
	vec3 emission;
	uint seed;
	uint depth;
};

struct Vertex {
	vec3 pos;
	vec3 normal;
	vec3 color;
	vec2 texCoord;
};

struct Material
{
	vec3 diffuse;
	vec3 emission;
};

struct Model
{
	uint64_t vertexAddress;
	uint64_t indexAddress;
	uint64_t matAddress;
	uint64_t matIndexAddress;
};

struct PushConstantRay {
	int frame;
};