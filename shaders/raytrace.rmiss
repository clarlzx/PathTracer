#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "raycommon.glsl"

layout(location = 0) rayPayloadInEXT HitPayload hitPayload;

void main()
{
    hitPayload.hitValue = vec3(0.0);
    hitPayload.emission = vec3(0.0);
    hitPayload.depth = 100;
}