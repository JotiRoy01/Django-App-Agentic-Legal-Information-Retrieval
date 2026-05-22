from django.db import models


class QueryRun(models.Model):

    class Status(models.TextChoices):
        PENDING = "PENDING", "Pending"
        STARTED = "STARTED", "Running"
        SUCCESS = "SUCCESS", "Complete"
        FAILURE = "FAILURE", "Failed"

    # Core fields
    query_text     = models.TextField(help_text="Raw English legal query")
    expanded_query = models.TextField(blank=True)
    task_id        = models.CharField(max_length=255, unique=True, null=True, blank=True,db_index=True, default=None)
    status         = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)

    # Results
    citations_json = models.TextField(blank=True)
    results_json   = models.TextField(blank=True)

    # Evaluation
    f1_score        = models.FloatField(null=True, blank=True)
    precision_score = models.FloatField(null=True, blank=True)
    recall_score    = models.FloatField(null=True, blank=True)

    # Pipeline settings
    hybrid_top_k   = models.IntegerField(default=100)   # fixed typo: was hybrid_tor_k
    reranker_top_k = models.IntegerField(default=10)
    use_stage2     = models.BooleanField(default=True)

    # Timing
    created_at   = models.DateTimeField(auto_now_add=True)   # fixed: was False
    completed_at = models.DateTimeField(null=True, blank=True)
    elapsed_secs = models.FloatField(null=True, blank=True)

    # Error
    error_message = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Query Run"

    def __str__(self):
        return f"[{self.status}] {self.query_text[:60]}"

    @property
    def is_complete(self):
        return self.status == self.Status.SUCCESS

    @property
    def is_failed(self):
        return self.status == self.Status.FAILURE

    @property
    def duration_display(self):
        if self.elapsed_secs is None:
            return "—"
        mins = int(self.elapsed_secs // 60)
        secs = int(self.elapsed_secs % 60)
        return f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
