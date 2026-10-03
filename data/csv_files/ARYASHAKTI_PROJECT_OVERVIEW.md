# AryaShakti Project Overview

Generated from the checked-in source by `build_project_overview.py`. This document is a static handover aid; live role permissions, environment values, external credentials, and database contents must be verified in the target environment.

## 1. What the project does
AryaShakti is a Django/ Django REST Framework backend for agricultural operations. It manages farmers, farms, organizations, crops, activities, subscriptions, payments, advisories, procurement, insurance, verification, notifications, and supporting admin workflows.

## 2. Technology stack
- Python with Django and Django REST Framework.
- PostgreSQL/PostGIS-style database configuration through `Project/settings.py`.
- Redis is declared in Docker Compose for asynchronous/background services.
- Gunicorn serves the API in the Docker image; WhiteNoise serves static assets.
- Celery, Django crontab, Firebase Admin, Razorpay, Google Pub/Sub, ONNX Runtime, pgvector, and cloud storage packages are present in the project dependencies.
- Swagger/ReDoc schemas are exposed from `Project/urls.py`.

## 3. Project structure
- `Project/`: settings, root URLs, WSGI, Celery, storage configuration.
- `app_farm/`: core farmers, farms, organizations, authentication, legacy APIs, and services.
- `recordActivity/`: activities, tasks, training, advisory embeddings, and semantic search.
- `payment/`, `subscription_service/`: payments, packages, subscriptions, and service balances.
- `verification/`, `insurance/`, `procurement/`: domain workflows and external integrations.
- `core/`: shared models, permissions, base views, and management commands.
- `libraries/`: integration clients and reusable service helpers.
- `templates/`, `static/`, `media/`, `logs/`: presentation, assets, uploaded/generated files, and logs.
- `Dockerfile`, `docker-compose.yml`, and `docker-compose2.yml`: container and local service definitions.

## 4. Main Django applications
The configured Django applications include:
- material
- django.contrib.admin
- django.contrib.auth
- django.contrib.contenttypes
- django.contrib.sessions
- django.contrib.messages
- django.contrib.staticfiles
- django.contrib.gis
- app_farm
- verification
- onemoney
- all_requests
- pop_manage
- cms
- certification_card
- farm_category
- subscription_service
- payment
- survey
- core
- broadcast
- insurance
- drf_yasg
- rest_framework
- storages
- import_export
- rest_framework.authtoken
- recordActivity
- django_crontab
- django_json_widget
- flat_json_widget
- procurement
- django_ckeditor_5
- admin_auto_filters

Model counts statically detected by app:
- `all_requests`: 1 model classes
- `app_farm`: 68 model classes
- `broadcast`: 4 model classes
- `certification_card`: 9 model classes
- `cms`: 5 model classes
- `core`: 7 model classes
- `farm_category`: 15 model classes
- `insurance`: 4 model classes
- `onemoney`: 2 model classes
- `payment`: 7 model classes
- `pop_manage`: 20 model classes
- `procurement`: 4 model classes
- `recordActivity`: 34 model classes
- `subscription_service`: 7 model classes
- `survey`: 4 model classes
- `verification`: 15 model classes

## 5. API architecture
- Root API prefixes are defined in `Project/urls.py`.
- `/blog/` -> `blog.urls`
- `/appapi/` -> `appapp.urls`
- `/api/` -> `app_farm.urls`
- `/api/v1/` -> `app_farm.routes`
- `/api/v1/verify/` -> `verification.urls`
- `/api/v1/onemoney/` -> `onemoney.urls`
- `/api/v1/record/` -> `recordActivity.urls`
- `/api/v1/pop/` -> `pop_manage.urls`
- `/api/v1/requests/` -> `all_requests.urls`
- `/api/v1/cms/` -> `cms.urls`
- `/api/v1/farm_category/` -> `farm_category.urls`
- `/api/v1/broadcast/` -> `broadcast.urls`
- `/vision/` -> `vision.urls`
- `/api/v1/certification/` -> `certification_card.urls`
- `/api/v1/subscription_service/` -> `subscription_service.urls`
- `/api/v1/payment/` -> `payment.urls`
- `/api/v1/sponsor/` -> `sponsor.urls`
- `/api/v1/survey/` -> `survey.urls`
- `/api/v1/core/` -> `core.urls`
- `/api/v1/procurement/` -> `procurement.urls`
- `/api/v1/insurance/` -> `insurance.urls`
- `/dashboard/` -> `webdashboard.urls`
- `/ckeditor5/` -> `django_ckeditor_5.urls`
- `/__debug__/` -> `debug_toolbar.urls`
- Legacy APIs are generally under `/api/` and newer service APIs under `/api/v1/`.
- Views use `APIView`, DRF serializers/validators, shared `BaseView` classes, and direct function-based views.
- Route source files detected: 20; direct route declarations detected: 398.
- Swagger and ReDoc are available under `/swagger/` and `/redoc/` when enabled by deployment configuration.

## 6. Authentication
- The project-wide DRF default is `CustomTokenAuthentication`.
- Tokens are stored in the `AuthenticationToken` model and are normally sent as `Authorization: Token <key>`.
- `CustomTokenAuthentication` validates the token, checks active/deleted user state, and returns the user/token pair.
- Some legacy APIs use a `secret-key` query/body parameter instead of the DRF token flow.
- `BaseView` applies `CustomTokenAuthentication` and `IsAuthenticated`; `BaseViewWithoutAuthentication` is used for public flows such as OTP send/verify.
- API-level role permission enforcement is present in the data model but the check in `app_farm/authentication.py` is currently commented out; verify effective permissions before relying on role restrictions.

## 7. User roles
The main role names referenced by code are:
- `Sponsor`
- `Admin`
- `Farmer`
- `BD`
- `BDManager`
- `ZONAL_HEAD`

Roles are database-backed through `app_farm.user_roles`; actual role rows and permissions must be queried in the target database.

## 8. Important business flows
- OTP login: send OTP, verify OTP, update login/device state, and issue an `AuthenticationToken` for existing users.
- Signup/farmer creation: validate secret key and registration data, create a user with verified state, assign a role, and issue a token.
- Farm operations: create, list, update, filter, and administer farms by organization, location, crop, health, and stress state.
- Subscription/payment: process orders, create `UserSubscription`/`UserService` records, deduct service validity, and handle Razorpay callbacks.
- Activities/training: create and manage farm activities, tasks, subtasks, training, and execution records.
- Advisory search: create embeddings for crop/disease advisories and use pgvector similarity search.
- Verification/insurance/procurement: connect domain records to external validation, insurance, procurement, and document workflows.

## 9. Database architecture
- Database connection values are supplied through environment variables and configured in `Project/settings.py`.
- Django models and migrations define the application schema; PostgreSQL/PostGIS features are used by the project.
- `recordActivity` also contains pgvector fields/migrations for advisory embeddings.
- Key model areas include users/roles, organizations/farms, activities, payments/subscriptions, verification, procurement, insurance, CMS, and notifications.
- Example model classes detected: `all_requests.Requests`, `app_farm.crop_master`, `app_farm.crop_variety_master`, `app_farm.user_roles`, `app_farm.OrgType`, `app_farm.organizations`, `app_farm.language_master`, `app_farm.user`, `app_farm.SoilReport`, `broadcast.SmsTemplate`, `broadcast.SmsTemplateField`, `broadcast.CommunitiesShare`, `broadcast.NotificationLog`, `certification_card.Certification`, `certification_card.FarmerCertification`, `certification_card.Score`, `certification_card.ScoreFarmer`, `certification_card.ScoreGeneralFarmer`, `certification_card.Badge`, `certification_card.FarmerBadge`, `certification_card.ValueType`, `cms.Page`, `cms.Section`, `cms.SubSection`, `cms.LanguageContent`.

## 10. External integrations
- Razorpay
- Firebase / FCM
- Google Pub/Sub
- SMS / OTP providers
- ONNX sentence embeddings
- Azure / cloud storage
- OpenAI / generative AI
- PAMS / bank validation

Integration credentials and endpoint configuration are environment-driven; never place real values in this generated document or source control.

## 11. Background processing
- Celery tasks are configured in `Project/celery.py` and `libraries/celery_task.py`.
- Django crontab jobs are declared as `CRONJOBS` in `Project/settings.py`.
- Several views/services use in-process thread pools for notifications and parallel work.
- Pub/Sub listeners are available through the management-command/listener modules.
- Docker Compose declares a Django API service, Redis, and a Celery worker with concurrency 2.
- The Docker image starts migrations/collectstatic, a PMS listener process, and Gunicorn.
- Inspect `Project/settings.py`, `Project/celery.py`, `libraries/celery_task.py`, `crons/`, and `core/management/commands/` before changing schedules or worker counts.

## 12. File handling
- Static files are collected into `staticfiles/` and served/configured through WhiteNoise.
- Media and uploaded files use Django storage settings; `Project/storage_backends.py` configures cloud-backed storage support.
- Upload and document workflows are spread across verification, procurement, CMS, farm, and record activity modules.
- Review file size/type validation and external signed-URL behavior before changing upload APIs.

## 13. Error handling & logging
- DRF uses `app_farm.custom_exception.custom_exception_handler` as the configured exception handler.
- Many older views catch broad exceptions and return project-specific error payloads; inspect the endpoint before assuming standard DRF errors.
- Application logs are written to `logs/` in local/container configurations, while Kubernetes deployments primarily expose stdout/stderr to centralized logging.
- API responses commonly include status, message, data, and internal error codes.

## 14. Configuration / environment variables
The settings module reads these variable names without exposing their values:
- `ALLOWED_HOSTS`
- `APILAYER_SHORT_KEY`
- `APP_ENV`
- `ARYASHAKTI_APP_API_SECRET_KEY`
- `ARYASHAKTI_DB_ENGINE`
- `ARYASHAKTI_DB_HOST`
- `ARYASHAKTI_DB_NAME`
- `ARYASHAKTI_DB_PASSWORD`
- `ARYASHAKTI_DB_PORT`
- `ARYASHAKTI_DB_USER`
- `ARYA_AG_URL`
- `ARYA_PROFIN_URL`
- `ATMS_PRIVATE_KEY`
- `ATMS_URL`
- `AXIS_API_BASE_URL`
- `AXIS_BANK_KEY_EXPORT_PASSWORD`
- `AXIS_CERTS_PEM_CERT_PATH`
- `AXIS_CERTS_PEM_KEY_PATH`
- `AXIS_CERTS_PRIVATE_KEY`
- `AXIS_CLIENT_ID`
- `AXIS_CLIENT_SECRET`
- `AXIS_CORP_ACC_NO`
- `AXIS_CORP_CHANNEL_ID`
- `AXIS_CORP_CODE`
- `AXIS_KEY`
- `BANK_ACCOUNT_DOCUMENT_ID`
- `CLIENT_CERT_PATH`
- `CLIENT_KEY_PATH`
- `CORS_ALLOWED_ORIGINS`
- `CSRF_TRUSTED_ORIGINS`
- `DEBUG`
- `DISTANCE_VALUE_FOR_POSTGIS`
- `DOMS_URL`
- `FILE_PATH_MEDIA_ROOT`
- `FILE_PATH_MEDIA_URL`
- `FILE_PATH_STATIC_ROOT`
- `FILE_PATH_STATIC_URL`
- `FIREBASE_DATABASE_NAME`
- `FIREBASE_DATABASE_URL`
- `FYLLO_URL`
- `FYLLO_USER_NAME`
- `FYLLO_USER_PASSWORD`
- `GCP_GEO_API_KEY`
- `GCS_BUCKET_NAME`
- `GCS_PREFIX`
- `GCS_PRIVATE_BUCKET_NAME`
- `GCS_PRIVATE_PREFIX`
- `GDAL_DATA`
- `GOOGLE_APIS_SERVER_TOKEN`
- `GOOGLE_PUSH_NOTIFICATION_URL`
- `GOOGLE_TRANSLATE_API_KEY`
- `HUNDRED_MS_API_URL`
- `HUNDRED_MS_APP_ACCESS_KEY`
- `HUNDRED_MS_SECRET_KEY`
- `IDS_BD_USER_ROLE`
- `IDS_COUNTRY_INDIA`
- `IDS_FARMER_USER_ROLE`
- `IDS_FARMER_WITHOUT_FPO_USER_ROLE`
- `LANGUAGE_CODE`
- `NEOPERK_URL`
- `NOMS_API_CLIENT_ID`
- `NOMS_AUTH_KEY`
- `NOMS_TENANT_ID`
- `NOMS_URL`
- `ONEMONEY_APP_IDENTIFIER`
- `ONEMONEY_BASE_URL`
- `ONEMONEY_CLIENT_ID`
- `ONEMONEY_CLIENT_SECRET`
- `ONEMONEY_DOCUMENT_ID`
- `ONEMONEY_ORGANISATION_ID`
- `ONEMONEY_PRODUCT_ID`
- `ONEMONEY_WEBHOOK_SECRET`
- `OPENAI_SECRET`
- `OSGEO4W_ROOT`
- `OTP_VERIFY_TOKEN`
- `OTP_VERIFY_URL`
- `PAMS_BASE_URL`
- `PAMS_CLIENT_ID`
- `PAMS_PAYLOAD_HASHING_SALT`
- `PAMS_SECRET_KEY`
- `PAMS_TENANT_ID`
- `PAMS_TIMEOUT`
- `PATH`
- `PRAKSHEP_API_SECRET_KEY`
- `PRAKSHEP_DB_HOST`
- `PRAKSHEP_DB_NAME`
- `PRAKSHEP_DB_PASSWORD`
- `PRAKSHEP_DB_PORT`
- `PRAKSHEP_DB_USER`
- `PROJ_LIB`
- `RAZORPAYX_ACCOUNT_NUM`
- `RAZORPAYX_API_KEY`
- `RAZORPAYX_BASE_URL`
- `RAZORPAYX_SECRET_KEY`
- `RAZORPAY_API_KEY`
- `RAZORPAY_SECRET_KEY`
- `RAZORPAY_WEBHOOK_SECRET`
- `ROOT_URLCONF`
- `SERVER_CA_CERT_PATH`
- `SIGNZY_ADHAR_DOC_ID`
- `SIGNZY_ADHAR_VERIFY_ACCESS_TOKEN`
- `SIGNZY_ADHAR_VERIFY_BASE_URL`
- `SIGNZY_BASE_URL`
- `SIGNZY_CALLBACK_URL`
- `SIGNZY_CONTRACT_BASE_URL`
- `SIGNZY_GST_DOC_ID`
- `SIGNZY_PAN_DOC_ID`
- `SIGNZY_PASSWORD`
- `SIGNZY_SIGNZY_EMAIL`
- `SIGNZY_USERNAME`
- `SMS_API_KEY`
- `SMS_API_PASSWORD`
- `SMS_API_SENDER_ID`
- `SMS_API_USER_ID`
- `SMTP_EMAIL_FROM`
- `SMTP_EMAIL_HOST`
- `SMTP_EMAIL_HOST_PASSWORD`
- `SMTP_EMAIL_HOST_USER`
- `SMTP_EMAIL_PORT`
- `SMTP_EMAIL_TO`
- `SMTP_EMAIL_USE_TLS`
- `SOCIAL_POST_TAG_API_URL`
- `TIME_ZONE`
- `USE_I18N`
- `USE_L10N`
- `USE_TZ`

Use `.env`, deployment secret/config objects, or the approved configuration system. Do not commit credentials, tokens, certificates, or database passwords.

## 15. Important services
- `app_farm.services.auth`: OTP, signup, farmer creation, and authentication token flows.
- `app_farm.services.farmer`: farmer-facing role and farm services.
- `app_farm.services.organizations`: organization workflows.
- `libraries/`: SMS, Firebase, payment, storage, Pub/Sub, AI, and domain-provider clients.
- `recordActivity.vector_files.advisory_embeddings`: lazy-loaded ONNX embedding service.
- `core.views.BaseView`: shared authenticated API behavior.

## 16. Important APIs
- `POST /api/v1/auth/send-otp` and `POST /api/v1/auth/otp/verify`: login OTP and token issuance.
- `POST /api/v1/auth/signup` and `POST /api/v1/auth/create_farmer`: user creation flows.
- `/api/farm_list_admin/`: organization/admin farm listing.
- `/api/v1/payment/`: orders, subscriptions, Razorpay webhooks, and payment services.
- `/api/v1/record/`: activity, task, training, and advisory-related APIs.
- `/api/v1/verify/`: document, boundary, Aadhaar, and verification workflows.
- `/api/v1/farm_category/`, `/api/v1/insurance/`, `/api/v1/procurement/`, and `/api/v1/survey/`: domain modules exposed under versioned routes.

Confirm exact request schemas in the endpoint validator and serializer before testing. Legacy and versioned endpoints can have different token/secret-key requirements.

## 17. Typical request flow
1. The request enters a root URL prefix in `Project/urls.py`.
2. Django resolves the app route to a view or service API.
3. DRF authentication runs when the view inherits `BaseView` or declares authentication classes.
4. A validator checks query/body data.
5. The view queries or mutates Django models and may call an external service.
6. A serializer builds the response, or a project response helper returns the standard envelope.
7. Exceptions are handled by local `try/except` blocks or the configured DRF exception handler.
8. Logs go to application stdout/stderr and/or configured log files.

## 18. Things to understand first
- Read `Project/settings.py`, `Project/urls.py`, `core/views.py`, and `app_farm/authentication.py` first.
- Understand the difference between `/api/` legacy routes and `/api/v1/` service routes.
- Trace users, roles, organizations, farms, subscriptions, and `AuthenticationToken` together.
- Inspect validators and serializers beside every API; many business rules are implemented there rather than in models.
- Verify environment variables and external credentials before running payment, SMS, Firebase, storage, or Pub/Sub flows.
- Treat broad exception handlers and commented permission checks as review points.
- Query live role rows and permissions before concluding that a role can or cannot call an endpoint.
- Confirm deployment image/branch, Gunicorn worker count, database target, and migration state before debugging production-like behavior.
