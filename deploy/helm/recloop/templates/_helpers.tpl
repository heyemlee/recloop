{{/* Labels of every object; "component" names the service. */}}
{{- define "recloop.labels" -}}
app.kubernetes.io/part-of: recloop
app.kubernetes.io/managed-by: {{ .root.Release.Service }}
helm.sh/chart: {{ .root.Chart.Name }}-{{ .root.Chart.Version }}
{{ include "recloop.selector" . }}
{{- end -}}

{{- define "recloop.selector" -}}
app.kubernetes.io/name: {{ .name }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
{{- end -}}

{{- define "recloop.image" -}}
{{ .repository }}:{{ .tag }}
{{- end -}}

{{/* Environment shared by the Go services: PostgreSQL (credentials from recloop-infra's
     Secret, expanded by the kubelet), Redis, Redpanda, the ad_token keys. `pg` overrides the
     PostgreSQL address (billing goes through pg-proxy). */}}
{{- define "recloop.goEnv" -}}
{{- $e := .root.Values.endpoints -}}
- name: PGUSER
  valueFrom:
    secretKeyRef: {name: {{ $e.postgresSecret }}, key: username}
- name: PGPASSWORD
  valueFrom:
    secretKeyRef: {name: {{ $e.postgresSecret }}, key: password}
- name: DATABASE_URL
  value: "postgres://$(PGUSER):$(PGPASSWORD)@{{ .pg | default $e.postgres }}/{{ $e.postgresDatabase }}?sslmode=disable"
- name: REDIS_URL
  value: "redis://{{ $e.redis }}/0"
- name: KAFKA_BROKERS
  value: {{ $e.kafkaBrokers | quote }}
- name: AD_TOKEN_KEYS
  valueFrom:
    secretKeyRef: {name: recloop-ad-token, key: keys}
{{- end -}}

{{/* Extra environment from values (a map). */}}
{{- define "recloop.envMap" -}}
{{- range $k, $v := . }}
- name: {{ $k }}
  value: {{ $v | quote }}
{{- end }}
{{- end -}}

{{/* The Go images run as nobody with a read-only root file system (static binaries on
     scratch, as in compose). */}}
{{- define "recloop.goPodSecurity" -}}
runAsNonRoot: true
runAsUser: 65534
runAsGroup: 65534
seccompProfile:
  type: RuntimeDefault
{{- end -}}

{{- define "recloop.containerSecurity" -}}
allowPrivilegeEscalation: false
readOnlyRootFilesystem: true
capabilities:
  drop: [ALL]
{{- end -}}

{{- define "recloop.bundleVolume" -}}
- name: bundles
  hostPath:
    path: {{ .Values.bundle.nodePath }}
    type: Directory
{{- end -}}
