{{- define "recloop-infra.labels" -}}
app.kubernetes.io/part-of: recloop
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}
