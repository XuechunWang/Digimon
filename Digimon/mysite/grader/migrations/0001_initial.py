# Generated by Django 2.2.3 on 2019-07-28 16:53

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Question',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question_title', models.TextField(max_length=100000)),
                ('set', models.IntegerField(unique=True)),
                ('min_score', models.IntegerField()),
                ('max_score', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='Essay',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('content', models.TextField(max_length=100000)),
                ('score', models.IntegerField(blank=True, null=True)),
                ('report', models.TextField(max_length=100000)),
                ('question', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='grader.Question')),
            ],
        ),
    ]
