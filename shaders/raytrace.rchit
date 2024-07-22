#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "raycommon.glsl"
#include "random.glsl"

layout(buffer_reference, scalar) buffer Vertices {Vertex v[];};
layout(buffer_reference, scalar) buffer Indices {uint i[];};
layout(buffer_reference, scalar) buffer Materials {Material m[];};
layout(buffer_reference, scalar) buffer MatIndices {uint i[];};
layout(push_constant) uniform _PushConstantRay {PushConstantRay pcRay;};
layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;
layout(binding = 1, set = 1) buffer _Model {Model m[];} model;

layout(location = 0) rayPayloadInEXT HitPayload hitPayload;
layout(location = 1) rayPayloadEXT bool isShadowed;

hitAttributeEXT vec2 attribs;

void main()
{
	// Model data
	Model modelData = model.m[gl_InstanceCustomIndexEXT];
	Vertices vertices = Vertices(modelData.vertexAddress);
	Indices indices = Indices(modelData.indexAddress);	
	Materials materials = Materials(modelData.matAddress);
	MatIndices matIndices = MatIndices(modelData.matIndexAddress);

	Material material = materials.m[matIndices.i[gl_PrimitiveID]];

	Vertex v0 = vertices.v[indices.i[gl_PrimitiveID * 3]];
	Vertex v1 = vertices.v[indices.i[gl_PrimitiveID * 3 + 1]];
	Vertex v2 = vertices.v[indices.i[gl_PrimitiveID * 3 + 2]];

	// Interpolate to get hit information
	const vec3 barycentricCoords = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
	// Assume that vertex normal is facing the right side
	const vec3 localHitNormal = v0.normal * barycentricCoords.x + v1.normal * barycentricCoords.y + v2.normal * barycentricCoords.z;
	const vec3 localHitPos = v0.pos * barycentricCoords.x + v1.pos * barycentricCoords.y + v2.pos * barycentricCoords.z;

	const vec3 worldHitPos = vec3(gl_ObjectToWorldEXT * vec4(localHitPos, 1.0));
	const vec3 worldHitNormal = normalize(vec3(localHitNormal * gl_WorldToObjectEXT)); 

	// Lambertian reflection
	vec3 rayDirection = normalize(worldHitNormal + rnd_unit_vector(hitPayload.seed));
	
	hitPayload.rayOrigin = worldHitPos;
	hitPayload.rayDirection = rayDirection;
	hitPayload.emission = material.emission;
	hitPayload.hitValue = vec3(1.0) * material.diffuse;

	if (material.emission.x + material.emission.y + material.emission.z > 0.0) {
		hitPayload.depth = 100;
	}
}
